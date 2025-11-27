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

type 'a weak_dynarray = 'a Stdlib.Weak.t ref
(** A dynamic array of weak references. *)

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

type 'a safe_lazy = {
  mutable value : [ `Callback of unit -> 'a | `Value of 'a ];
  unique_id : string;
}
[@@deriving sexp_of]
(** A lazy value that can be safely forced and compared by unique ID. *)

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

(** A persistent map implemented as a balanced binary tree. The sexp_of function preserves and
    displays the tree structure. *)
module Tree_map = struct
  type ('k, 'v) t =
    | Empty
    | Node of { key : 'k; value : 'v; left : ('k, 'v) t; right : ('k, 'v) t; height : int }

  let empty = Empty
  let height = function Empty -> 0 | Node { height; _ } -> height

  let create key value left right =
    Node { key; value; left; right; height = 1 + max (height left) (height right) }

  let balance_factor = function Empty -> 0 | Node { left; right; _ } -> height left - height right

  let rotate_right = function
    | Node
        {
          key;
          value;
          left = Node { key = lkey; value = lvalue; left = ll; right = lr; _ };
          right;
          _;
        } ->
        create lkey lvalue ll (create key value lr right)
    | t -> t

  let rotate_left = function
    | Node
        {
          key;
          value;
          left;
          right = Node { key = rkey; value = rvalue; left = rl; right = rr; _ };
          _;
        } ->
        create rkey rvalue (create key value left rl) rr
    | t -> t

  let rebalance t =
    match balance_factor t with
    | bf when bf > 1 -> (
        match t with
        | Node { left; _ } when balance_factor left < 0 ->
            rotate_right
              (create
                 (match t with Node n -> n.key | _ -> assert false)
                 (match t with Node n -> n.value | _ -> assert false)
                 (rotate_left left)
                 (match t with Node n -> n.right | _ -> assert false))
        | _ -> rotate_right t)
    | bf when bf < -1 -> (
        match t with
        | Node { right; _ } when balance_factor right > 0 ->
            rotate_left
              (create
                 (match t with Node n -> n.key | _ -> assert false)
                 (match t with Node n -> n.value | _ -> assert false)
                 (match t with Node n -> n.left | _ -> assert false)
                 (rotate_right right))
        | _ -> rotate_left t)
    | _ -> t

  let rec add ~compare ~key ~data t =
    match t with
    | Empty -> create key data Empty Empty
    | Node n ->
        let c = compare key n.key in
        if c = 0 then create key data n.left n.right
        else if c < 0 then rebalance (create n.key n.value (add ~compare ~key ~data n.left) n.right)
        else rebalance (create n.key n.value n.left (add ~compare ~key ~data n.right))

  let rec find ~compare ~key t =
    match t with
    | Empty -> None
    | Node n ->
        let c = compare key n.key in
        if c = 0 then Some n.value
        else if c < 0 then find ~compare ~key n.left
        else find ~compare ~key n.right

  let rec mem ~compare ~key t =
    match t with
    | Empty -> false
    | Node n ->
        let c = compare key n.key in
        if c = 0 then true
        else if c < 0 then mem ~compare ~key n.left
        else mem ~compare ~key n.right

  let rec fold t ~init ~f =
    match t with
    | Empty -> init
    | Node n ->
        let acc = fold n.left ~init ~f in
        let acc = f ~key:n.key ~data:n.value acc in
        fold n.right ~init:acc ~f

  let rec iter t ~f =
    match t with
    | Empty -> ()
    | Node n ->
        iter n.left ~f;
        f ~key:n.key ~data:n.value;
        iter n.right ~f

  let rec map t ~f =
    match t with
    | Empty -> Empty
    | Node n -> create n.key (f n.value) (map n.left ~f) (map n.right ~f)

  let rec mapi t ~f =
    match t with
    | Empty -> Empty
    | Node n -> create n.key (f ~key:n.key ~data:n.value) (mapi n.left ~f) (mapi n.right ~f)

  let to_alist t = List.rev (fold t ~init:[] ~f:(fun ~key ~data acc -> (key, data) :: acc))

  let of_alist ~compare lst =
    List.fold lst ~init:Empty ~f:(fun acc (key, data) -> add ~compare ~key ~data acc)

  (** Sexp conversion that preserves tree structure *)
  let rec sexp_of_t sexp_of_k sexp_of_v = function
    | Empty -> Sexp.Atom "Empty"
    | Node { key; value; left; right; _ } ->
        Sexp.List
          [
            Sexp.Atom "Node";
            Sexp.List [ Sexp.Atom "key"; sexp_of_k key ];
            Sexp.List [ Sexp.Atom "value"; sexp_of_v value ];
            Sexp.List [ Sexp.Atom "left"; sexp_of_t sexp_of_k sexp_of_v left ];
            Sexp.List [ Sexp.Atom "right"; sexp_of_t sexp_of_k sexp_of_v right ];
          ]
end
