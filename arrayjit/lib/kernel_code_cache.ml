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

let publish c ~keys ~bytes v =
  locked c (fun () ->
      let capacity = c.capacity_bytes () in
      shrink_to c (Int.max 0 capacity);
      if capacity > 0 && bytes <= capacity && not (List.is_empty keys) then (
        let id = c.next_id in
        c.next_id <- id + 1;
        List.iter keys ~f:(fun k -> Hashtbl.set c.entries ~key:k ~data:(id, v));
        Queue.enqueue c.groups { id; group_keys = keys; group_bytes = bytes };
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

let is_ident_start c = Char.is_alpha c || Char.equal c '_'
let is_ident_char c = Char.is_alphanum c || Char.equal c '_'

(* The spellings the lowering mints fresh: a symbol [i<N>] ([Indexing.symbol_ident]) and a
   scope-local [v<N>_<ident>] ([C_syntax]'s local names). The suffix of a local is kept. *)
let minted_spelling tok =
  let n = String.length tok in
  let digits_end =
    let j = ref 1 in
    while !j < n && Char.is_digit tok.[!j] do
      Int.incr j
    done;
    !j
  in
  if n < 2 || digits_end = 1 then None
  else
    match tok.[0] with
    | 'i' when digits_end = n -> Some ""
    | 'v' when digits_end < n && Char.equal tok.[digits_end] '_' ->
        Some (String.drop_prefix tok digits_end)
    | _ -> None

let canonical_symbols ?(keep = []) text =
  let n = String.length text in
  let out = Buffer.create (n + 64) in
  let names = Hashtbl.create (module String) in
  let canonical_of tok suffix =
    Hashtbl.find_or_add names tok ~default:(fun () ->
        "ocannl_k" ^ Int.to_string (Hashtbl.length names) ^ suffix)
  in
  let at i s =
    i + String.length s <= n && String.equal (String.sub text ~pos:i ~len:(String.length s)) s
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
  (* [in_comment]: [`Block] until [*/], [`Line] until a newline; a quote there is prose, not a
     literal, and identifiers are renamed as in code. *)
  let rec go ~in_comment i =
    if i < n then
      let c = text.[i] in
      match in_comment with
      | `Block when at i "*/" ->
          Buffer.add_string out "*/";
          go ~in_comment:`Code (i + 2)
      | `Line when Char.equal c '\n' ->
          Buffer.add_char out c;
          go ~in_comment:`Code (i + 1)
      | `Code when at i "/*" ->
          Buffer.add_string out "/*";
          go ~in_comment:`Block (i + 2)
      | `Code when at i "//" ->
          Buffer.add_string out "//";
          go ~in_comment:`Line (i + 2)
      | `Code when Char.equal c '"' || Char.equal c '\'' ->
          Buffer.add_char out c;
          go ~in_comment (literal c (i + 1))
      | _ when is_ident_start c ->
          let j = ref (i + 1) in
          while !j < n && is_ident_char text.[!j] do
            Int.incr j
          done;
          let tok = String.sub text ~pos:i ~len:(!j - i) in
          Buffer.add_string out
            (match minted_spelling tok with
            | Some suffix when not (List.mem keep tok ~equal:String.equal) ->
                canonical_of tok suffix
            | Some _ | None -> tok);
          go ~in_comment !j
      | _ when Char.is_digit c ->
          (* A number, suffix and exponent letters included, is never an identifier. *)
          let j = ref (i + 1) in
          while !j < n && (is_ident_char text.[!j] || Char.equal text.[!j] '.') do
            Int.incr j
          done;
          Buffer.add_string out (String.sub text ~pos:i ~len:(!j - i));
          go ~in_comment !j
      | _ ->
          Buffer.add_char out c;
          go ~in_comment (i + 1)
  in
  go ~in_comment:`Code 0;
  Buffer.contents out
