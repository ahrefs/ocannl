open Base

module Set_O = struct
  let ( + ) = Set.union
  let ( - ) = Set.diff
  let ( & ) = Set.inter

  let ( -* ) s1 s2 =
    Set.of_sequence (Set.comparator_s s1)
    @@ Sequence.map ~f:Either.value @@ Set.symmetric_diff s1 s2
end

let no_ints = Set.empty (module Int)
let one_int = Set.singleton (module Int)

let map_merge m1 m2 ~f =
  Map.merge m1 m2 ~f:(fun ~key:_ m ->
      match m with `Right v | `Left v -> Some v | `Both (v1, v2) -> Some (f v1 v2))

let mref_add mref ~key ~data ~or_ =
  match Map.add !mref ~key ~data with
  | `Ok m -> mref := m
  | `Duplicate -> or_ (Map.find_exn !mref key)

let mref_add_missing mref key ~f =
  if Map.mem !mref key then () else mref := Map.add_exn !mref ~key ~data:(f ())

(** A mutable linked list structure. *)
type 'a mutable_list = Empty | Cons of { hd : 'a; mutable tl : 'a mutable_list }
[@@deriving equal, sexp, variants]

let insert ~next = function
  | Empty -> Cons { hd = next; tl = Empty }
  | Cons cons ->
      cons.tl <- Cons { hd = next; tl = cons.tl };
      cons.tl

let tl_exn = function
  | Empty -> raise @@ Not_found_s (Sexp.Atom "mutable_list.tl_exn")
  | Cons { tl; _ } -> tl

(** A dynamic array of weak references. *)
type 'a weak_dynarray = 'a Stdlib.Weak.t ref

let weak_create () : 'a weak_dynarray = ref @@ Stdlib.Weak.create 0

let sexp_of_weak_dynarray sexp_of_elem arr =
  sexp_of_array (sexp_of_option sexp_of_elem) Stdlib.Weak.(Array.init (length !arr) ~f:(get !arr))

let register_new (arr : 'a weak_dynarray) ?(grow_by = 1) create =
  let module W = Stdlib.Weak in
  let old = !arr in
  let pos = ref 0 in
  while !pos < W.length old && W.check old !pos do
    Int.incr pos
  done;
  if !pos >= W.length old then (
    arr := Stdlib.Weak.create (W.length old + grow_by);
    Stdlib.Weak.blit old 0 !arr 0 (Stdlib.Weak.length old));
  let v = create !pos in
  W.set !arr !pos (Some v);
  v

let weak_iter (arr : 'a weak_dynarray) ~f =
  let module W = Stdlib.Weak in
  for i = 0 to W.length !arr - 1 do
    Option.iter (W.get !arr i) ~f
  done

(** A lazy value that can be safely forced and compared by unique ID. *)
type 'a safe_lazy = {
  mutable value : [ `Callback of unit -> 'a | `Value of 'a ];
  unique_id : string;
}
[@@deriving sexp_of]

let safe_lazy unique_id f = { value = `Callback f; unique_id }

let safe_force gated =
  match gated.value with
  | `Value v -> v
  | `Callback f ->
      let v = f () in
      gated.value <- `Value v;
      v

let is_safe_val = function { value = `Value _; _ } -> true | _ -> false

let safe_map ~upd ~f gated =
  let unique_id = gated.unique_id ^ "_" ^ upd in
  match gated.value with
  | `Value v -> { value = `Value (f v); unique_id }
  | `Callback callback -> { value = `Callback (fun () -> f (callback ())); unique_id }

let equal_safe_lazy equal_elem g1 g2 =
  match (g1.value, g2.value) with
  | `Value v1, `Value v2 ->
      (* Both values are forced - assert uniqueness *)
      let id_equal = String.equal g1.unique_id g2.unique_id in
      if id_equal then assert (equal_elem v1 v2);
      id_equal
  | _ -> String.equal g1.unique_id g2.unique_id

let compare_safe_lazy compare_elem g1 g2 =
  match (g1.value, g2.value) with
  | `Value v1, `Value v2 ->
      (* Both values are forced - assert uniqueness *)
      let id_cmp = String.compare g1.unique_id g2.unique_id in
      if id_cmp = 0 then assert (compare_elem v1 v2 = 0);
      id_cmp
  | _ -> String.compare g1.unique_id g2.unique_id

let hash_fold_safe_lazy _hash_elem state gated = hash_fold_string state gated.unique_id

let sexp_of_safe_lazy sexp_of_elem gated =
  let status =
    match gated.value with `Callback _ -> Sexp.Atom "pending" | `Value v -> sexp_of_elem v
  in
  Sexp.List
    [
      Sexp.Atom "safe_lazy";
      Sexp.List [ Sexp.Atom "id"; Sexp.Atom gated.unique_id ];
      Sexp.List [ Sexp.Atom "value"; status ];
    ]
