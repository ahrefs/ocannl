open Base

type group = { id : int; group_keys : string list; group_bytes : int }

type 'v t = {
  capacity_bytes : unit -> int;
  entries : (string, int * 'v) Hashtbl.t;  (** key -> owning group id, value *)
  groups : group Queue.t;  (** oldest first *)
  mutable retained_bytes : int;
  mutable next_id : int;
  mutable evicted_groups : int;
  lock : Stdlib.Mutex.t;
}

type stats = { n_groups : int; n_entries : int; n_bytes : int; n_evicted_groups : int }

let create ~capacity_bytes =
  {
    capacity_bytes;
    entries = Hashtbl.create (module String);
    groups = Queue.create ();
    retained_bytes = 0;
    next_id = 0;
    evicted_groups = 0;
    lock = Stdlib.Mutex.create ();
  }

let key ~parts =
  let b = Buffer.create 1024 in
  List.iter parts ~f:(fun p ->
      Buffer.add_string b (Int.to_string (String.length p));
      Buffer.add_char b ':';
      Buffer.add_string b p);
  Stdlib.Digest.to_hex (Stdlib.Digest.string (Buffer.contents b))

let locked c f =
  Stdlib.Mutex.lock c.lock;
  Exn.protect ~f ~finally:(fun () -> Stdlib.Mutex.unlock c.lock)

let enabled c = c.capacity_bytes () > 0

let evict_oldest c =
  match Queue.dequeue c.groups with
  | None -> ()
  | Some g ->
      List.iter g.group_keys ~f:(fun k ->
          match Hashtbl.find c.entries k with
          | Some (owner, _) when owner = g.id -> Hashtbl.remove c.entries k
          | Some _ | None -> ());
      c.retained_bytes <- c.retained_bytes - g.group_bytes;
      c.evicted_groups <- c.evicted_groups + 1

let shrink_to c capacity =
  while c.retained_bytes > capacity && not (Queue.is_empty c.groups) do
    evict_oldest c
  done

let find c k =
  locked c (fun () ->
      let capacity = c.capacity_bytes () in
      shrink_to c (Int.max 0 capacity);
      if capacity <= 0 then None else Option.map (Hashtbl.find c.entries k) ~f:snd)

let publish c ~entries ~bytes =
  locked c (fun () ->
      let capacity = c.capacity_bytes () in
      shrink_to c (Int.max 0 capacity);
      if capacity > 0 && bytes <= capacity && not (List.is_empty entries) then (
        let id = c.next_id in
        c.next_id <- id + 1;
        List.iter entries ~f:(fun (k, v) -> Hashtbl.set c.entries ~key:k ~data:(id, v));
        Queue.enqueue c.groups { id; group_keys = List.map entries ~f:fst; group_bytes = bytes };
        c.retained_bytes <- c.retained_bytes + bytes;
        shrink_to c capacity))

let stats c =
  locked c (fun () ->
      {
        n_groups = Queue.length c.groups;
        n_entries = Hashtbl.length c.entries;
        n_bytes = c.retained_bytes;
        n_evicted_groups = c.evicted_groups;
      })

(* {2 Concurrent programs (gh-ocannl-1603)} *)

type split = { max_programs : int; min_kernels : int }

let programs_for split n =
  if split.max_programs <= 1 || n <= 1 then 1
  else Int.max 1 (Int.min (Int.min split.max_programs n) (n / Int.max 1 split.min_kernels))

let chunks k idxs =
  let a = Array.of_list idxs in
  let n = Array.length a in
  List.init k ~f:(fun t ->
      let lo = t * n / k and hi = (t + 1) * n / k in
      Array.to_list (Array.sub a ~pos:lo ~len:(hi - lo)))
  |> List.filter ~f:(fun chunk -> not (List.is_empty chunk))

let split_override : split option ref = ref None
let current_split_override () = !split_override

let with_split split f =
  let saved = !split_override in
  split_override := Some split;
  Exn.protect ~f ~finally:(fun () -> split_override := saved)

let split_capable_count = Stdlib.Atomic.make 0
let concurrent_program_count = Stdlib.Atomic.make 0
let record_split_capable () = Stdlib.Atomic.incr split_capable_count
let split_capable_compiles () = Stdlib.Atomic.get split_capable_count
let programs_compiled_concurrently () = Stdlib.Atomic.get concurrent_program_count

type ('p, 'w) concurrent = {
  split : split;
  prepare : group:int -> n_groups:int -> string list -> int list -> unit -> 'w;
  finish : group:int -> 'w -> 'p * int;
  reraise : 'a. group:int -> int list -> exn -> Stdlib.Printexc.raw_backtrace -> 'a;
  phase : (unit -> unit) -> unit;
}

(* Runs every thunk at once: the first on the calling domain, each other on a domain of its own (or,
   when no domain can be spawned, on the calling domain after the first). Every one is joined before
   this returns, and none raises out of it: a failure comes back as its exception and backtrace. *)
let run_all thunks =
  let capture f = try Ok (f ()) with exn -> Error (exn, Stdlib.Printexc.get_raw_backtrace ()) in
  match thunks with
  | [] -> []
  | first :: rest ->
      let spawned =
        List.map rest ~f:(fun f ->
            match Stdlib.Domain.spawn (fun () -> capture f) with
            | d -> `Spawned d
            | exception _ -> `Inline f)
      in
      let first = capture first in
      first
      :: List.map spawned ~f:(function
        | `Spawned d -> Stdlib.Domain.join d
        | `Inline f -> capture f)

(* {2 Lookup, dedup and compile of the misses} *)

let resolve c ~lock ~keys ~options ~symbols ?(always_lock = false)
    ?(before_compile = fun ~n_compiled:_ ~n_groups:_ -> ()) ?concurrent ~compile_group () =
  let n = Array.length keys in
  let found = Array.create ~len:n None in
  let all = List.init n ~f:Fn.id in
  let lookup i = if Option.is_none found.(i) then found.(i) <- find c keys.(i) in
  let misses () = List.filter all ~f:(fun i -> Option.is_none found.(i)) in
  (* Without the lock first, so an all-hit compile never waits on another domain's compiler; the
     misses are looked up again under it, where another domain may just have published them. *)
  List.iter all ~f:lookup;
  let compile_misses () =
    List.iter (misses ()) ~f:lookup;
    let misses = misses () in
    (* One representative per distinct key, the first in batch order; the others share its code. *)
    let rep_of_key = Hashtbl.create (module String) in
    let reps =
      List.filter misses ~f:(fun i ->
          match Hashtbl.find rep_of_key keys.(i) with
          | Some _ -> false
          | None ->
              Hashtbl.set rep_of_key ~key:keys.(i) ~data:i;
              true)
    in
    (* Representatives grouped by option list, in batch order: one program per distinct list, so a
       kernel's code always comes from its own option list. *)
    let same = List.equal String.equal in
    let distinct =
      List.fold reps ~init:[] ~f:(fun acc i ->
          if List.mem acc options.(i) ~equal:same then acc else acc @ [ options.(i) ])
    in
    let groups =
      List.map distinct ~f:(fun opts ->
          (opts, List.filter reps ~f:(fun i -> same options.(i) opts)))
    in
    let publish_group idxs (program, bytes) =
      let entries = List.map idxs ~f:(fun i -> (keys.(i), (program, symbols.(i)))) in
      (* Published only now, after the compiler returned: a failed group leaves no entry. *)
      publish c ~entries ~bytes;
      List.iter idxs ~f:(fun i -> found.(i) <- Some (program, symbols.(i)))
    in
    (* gh-ocannl-1603: with [concurrent], a group of enough representatives is split into several
       programs of contiguous representatives; when any group splits, every program of the batch is
       compiled at once. *)
    let programs =
      match concurrent with
      | None -> None
      | Some cc ->
          let programs =
            List.concat_map groups ~f:(fun (opts, idxs) ->
                List.map
                  (chunks (programs_for cc.split (List.length idxs)) idxs)
                  ~f:(fun chunk -> (opts, chunk)))
          in
          if List.length programs > List.length groups then Some (cc, programs) else None
    in
    (match programs with
    | None ->
        before_compile ~n_compiled:(List.length reps) ~n_groups:(List.length groups);
        List.iteri groups ~f:(fun k (opts, idxs) ->
            publish_group idxs (compile_group ~group:k ~n_groups:(List.length groups) opts idxs))
    | Some (cc, programs) ->
        let n_groups = List.length programs in
        before_compile ~n_compiled:(List.length reps) ~n_groups;
        (* Prepared in batch order on this domain, before any compile starts. *)
        let works =
          List.mapi programs ~f:(fun k (opts, idxs) -> cc.prepare ~group:k ~n_groups opts idxs)
        in
        ignore (Stdlib.Atomic.fetch_and_add concurrent_program_count n_groups : int);
        let results = ref [] in
        cc.phase (fun () -> results := run_all works);
        (* Every compile has returned: each one that succeeded is finished and published in batch
           order on this domain, then the first failure, in batch order, is re-raised here. *)
        let failure =
          List.foldi (List.zip_exn programs !results) ~init:None
            ~f:(fun k failure ((_, idxs), result) ->
              match result with
              | Ok work ->
                  publish_group idxs (cc.finish ~group:k work);
                  failure
              | Error (exn, backtrace) -> Option.first_some failure (Some (k, idxs, exn, backtrace)))
        in
        Option.iter failure ~f:(fun (k, idxs, exn, backtrace) ->
            cc.reraise ~group:k idxs exn backtrace));
    List.iter misses ~f:(fun i ->
        if Option.is_none found.(i) then found.(i) <- found.(Hashtbl.find_exn rep_of_key keys.(i)))
  in
  if List.is_empty (misses ()) then (if always_lock then Stdlib.Mutex.protect lock compile_misses)
  else Stdlib.Mutex.protect lock compile_misses;
  Array.map found ~f:(fun v -> Option.value_exn ~here:[%here] v)

(* {2 The kernel's canonical text} *)

let segment_kernel_name ~routine i = routine ^ "__seg" ^ Int.to_string i

let kernel_token name =
  match String.substr_index_all name ~may_overlap:false ~pattern:"__seg" with
  | [] -> name
  | idxs ->
      let j = List.last_exn idxs in
      let digits = String.drop_prefix name (j + 5) in
      if (not (String.is_empty digits)) && String.for_all digits ~f:Char.is_digit then
        String.prefix name (j + 5)
      else name

let is_ident_start c = Char.is_alpha c || Char.equal c '_'
let is_ident_char c = Char.is_alphanum c || Char.equal c '_'
let all_digits s = (not (String.is_empty s)) && String.for_all s ~f:Char.is_digit

(* A loop or static-index symbol as [Indexing.symbol_ident] spells it: [i] then digits. *)
let is_symbol tok =
  String.length tok >= 2 && Char.equal tok.[0] 'i' && all_digits (String.drop_prefix tok 1)

(* A scope-local as [C_syntax] spells it: [v], digits, [_], anything. *)
let is_scope_local tok =
  String.length tok >= 3
  && Char.equal tok.[0] 'v'
  &&
  match String.index_from tok 1 '_' with
  | Some j -> j > 1 && all_digits (String.sub tok ~pos:1 ~len:(j - 1))
  | None -> false

let canonical_kernel ?kernel_name ?(keep = []) ?(bound = []) ?(derivations = []) ?(free = []) text =
  let free = Hash_set.of_list (module String) free in
  let keep = Hash_set.of_list (module String) keep in
  let bound =
    (* A declared name that is also a free name of the language, a builtin, a vendor identifier or a
       predefined macro -- or that is spelled in the implementation's reserved namespace ([__x],
       [_X]), where a compiler's own macros live -- is renamed nowhere: its occurrences might be the
       free meaning, and a macro is not shadowed by a declaration, so such a kernel may not compile
       at all, and its key must not reach the code of a twin that did (gh-ocannl-1525 review). *)
    let reserved b =
      String.is_prefix b ~prefix:"__"
      || (String.length b >= 2 && Char.equal b.[0] '_' && Char.is_uppercase b.[1])
    in
    if List.exists bound ~f:(fun b -> Hash_set.mem free b || reserved b) then
      Hash_set.create (module String)
    else Hash_set.of_list (module String) bound
  in
  let is_base base = Hash_set.mem bound base || is_symbol base || all_digits base in
  let derived tok =
    List.exists derivations ~f:(fun (prefix, suffix) ->
        String.length tok > String.length prefix + String.length suffix
        && String.is_prefix tok ~prefix && String.is_suffix tok ~suffix
        && is_base
             (String.sub tok ~pos:(String.length prefix)
                ~len:(String.length tok - String.length prefix - String.length suffix)))
  in
  let renamed tok =
    (not (Hash_set.mem keep tok))
    && (not (Hash_set.mem free tok))
    && (is_symbol tok || is_scope_local tok || Hash_set.mem bound tok || derived tok)
  in
  let n = String.length text in
  let out = Buffer.create (n + 64) in
  let names = Hashtbl.create (module String) in
  let canonical_of tok =
    Hashtbl.find_or_add names tok ~default:(fun () ->
        "ocannl_k" ^ Int.to_string (Hashtbl.length names))
  in
  let at i s =
    i + String.length s <= n && String.equal (String.sub text ~pos:i ~len:(String.length s)) s
  in
  (* Copies verbatim from [i] through the end of [stop] (or of the text). *)
  let copy_through i stop =
    let rec find j =
      if j >= n then n else if at j stop then j + String.length stop else find (j + 1)
    in
    let j = find i in
    Buffer.add_string out (String.sub text ~pos:i ~len:(j - i));
    j
  in
  (* Copies a quoted literal starting at [i] (the opening quote) through its closing quote. *)
  let rec literal quote i =
    if i >= n then i
    else
      let c = text.[i] in
      Buffer.add_char out c;
      if Char.equal c '\\' && i + 1 < n then (
        Buffer.add_char out text.[i + 1];
        literal quote (i + 2))
      else if Char.equal c quote then i + 1
      else literal quote (i + 1)
  in
  (* A preprocessor line, continuations included, is copied verbatim. *)
  let rec directive i =
    if i >= n then i
    else if Char.equal text.[i] '\\' && i + 1 < n && Char.equal text.[i + 1] '\n' then (
      Buffer.add_string out "\\\n";
      directive (i + 2))
    else if Char.equal text.[i] '\n' then i
    else (
      Buffer.add_char out text.[i];
      directive (i + 1))
  in
  (* [line_start]: only blanks since the last newline, where a [#] opens a directive. [member]: the
     last significant token was [.], [->] or [::], so an identifier here names a member or a
     qualified name -- never a binding of the kernel -- and is copied as is. *)
  let rec go ~in_comment ~line_start ~member i =
    if i < n then
      let c = text.[i] in
      match in_comment with
      | `Block when at i "*/" ->
          Buffer.add_string out "*/";
          go ~in_comment:`Code ~line_start:false ~member:false (i + 2)
      | `Line when Char.equal c '\n' ->
          Buffer.add_char out c;
          go ~in_comment:`Code ~line_start:true ~member:false (i + 1)
      | `Code when Char.equal c '\n' ->
          Buffer.add_char out c;
          go ~in_comment ~line_start:true ~member (i + 1)
      | _ when Char.equal c ' ' || Char.equal c '\t' || Char.equal c '\r' ->
          Buffer.add_char out c;
          go ~in_comment ~line_start ~member (i + 1)
      | `Code when line_start && Char.equal c '#' ->
          go ~in_comment ~line_start:false ~member:false (directive i)
      | `Code when at i "/*" ->
          Buffer.add_string out "/*";
          go ~in_comment:`Block ~line_start:false ~member:false (i + 2)
      | `Code when at i "//" ->
          Buffer.add_string out "//";
          go ~in_comment:`Line ~line_start:false ~member:false (i + 2)
      | `Code when at i "[[" ->
          (* An attribute ([[buffer(0)]], [[thread_position_in_grid]]): fixed vocabulary. *)
          go ~in_comment ~line_start:false ~member:false (copy_through i "]]")
      | `Code when Char.equal c '"' || Char.equal c '\'' ->
          Buffer.add_char out c;
          go ~in_comment ~line_start:false ~member:false (literal c (i + 1))
      | _ when at i "->" || at i "::" ->
          Buffer.add_string out (String.sub text ~pos:i ~len:2);
          go ~in_comment ~line_start:false ~member:true (i + 2)
      | _ when Char.equal c '.' ->
          Buffer.add_char out c;
          go ~in_comment ~line_start:false ~member:true (i + 1)
      | _ when is_ident_start c ->
          let j = ref (i + 1) in
          while !j < n && is_ident_char text.[!j] do
            Int.incr j
          done;
          let tok = String.sub text ~pos:i ~len:(!j - i) in
          Buffer.add_string out
            (if member then tok
             else if Option.exists kernel_name ~f:(String.equal tok) then kernel_token tok
             else if renamed tok then canonical_of tok
             else tok);
          go ~in_comment ~line_start:false ~member:false !j
      | _ when Char.is_digit c ->
          (* A number, suffix, exponent and decimal point included, is never an identifier. *)
          let j = ref (i + 1) in
          while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
            Int.incr j
          done;
          Buffer.add_string out (String.sub text ~pos:i ~len:(!j - i));
          go ~in_comment ~line_start:false ~member:false !j
      | _ ->
          Buffer.add_char out c;
          go ~in_comment ~line_start:false ~member:false (i + 1)
  in
  go ~in_comment:`Code ~line_start:true ~member:false 0;
  Buffer.contents out
