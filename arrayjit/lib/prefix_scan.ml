(* The prefix-scan rewrite (gh-ocannl-1758); the contract is in the interface. *)

open Base
module LL = Low_level
module Tn = Tnode
module Idx = Indexing
open Lowered_nests

let override : bool option ref = ref None
let set_enabled b = override := b

let enabled () =
  match !override with
  | Some b -> b
  | None -> Utils.get_global_flag ~default:false ~arg_name:"prefix_scan"

(* {1 Minted state}

   One carried scalar per rewritten nest, in a namespace of its own (the online softmax's [rewrite]
   namespace numbers its nodes from its own counter). Sibling lowerings of one program must mint the
   SAME nodes, or the analysis cache misses on every one of them, so a state node is memoized by the
   matched site: the target node, the match's ordinal among the scans into that target, and the
   precision -- never by the target alone, which would make two scans accumulating into one gradient
   share a carried local. *)

let namespace = "prefix_scan"
let provenance = Tn.Site "1758:prefix-scan-state"

let fresh_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

module Site_key = struct
  type t = int * int * string [@@deriving compare, hash, sexp_of]
end

let minted : (Site_key.t, Tn.t) Hashtbl.t Tn.Weak_table.t = Tn.Weak_table.create 16

(* State node uid to the uid of the target it accumulates into: the census of scans already in a
   routine, so that a later fixpoint round numbers new sites after them. Keyed by the state node, so
   it lasts as long as the node: the reset must not orphan a state already in some lowered code, and
   a dropped state's goes with it (gh-ocannl-1859). *)
let owners : int Tn.Weak_table.t = Tn.Weak_table.create 16
let reset () = Tn.Weak_table.reset minted
let retained () = Tn.live_entries minted + Tn.live_entries owners
let () = Tn.before_accessibility_snapshot := reset :: !Tn.before_accessibility_snapshot
let state_owner (tn : Tn.t) = Tn.Weak_table.find_opt owners tn

(* The carried state's precision: the target's widened to f32, f64 kept. A target that is not a
   float declines: an f32 state would round an integer sum the composed reduction holds exactly. *)
let state_prec_of_prec = function
  | Ops.Double_prec _ as p -> Some p
  | Ops.Single_prec _ | Ops.Half_prec _ | Ops.Bfloat16_prec _ | Ops.Fp8_prec _ -> Some Ops.single
  | _ -> None

let state_prec (tn : Tn.t) = state_prec_of_prec (Lazy.force tn.Tn.storage_prec)

(* [role] tells the carried values of one site apart: the gradient gate's scan carries the run's sum
   and the previous extremum, which must not share a node. *)
let state_node ?(role = "") ~(y : Tn.t) ~ordinal prec =
  Hashtbl.find_or_add
    (Tn.memo_bucket minted y ~create:(fun () -> Hashtbl.create (module Site_key)))
    (y.Tn.uid, ordinal, role ^ Ops.prec_string prec)
    ~default:(fun () ->
      let tn =
        Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ())
          ~label:((if String.is_empty role then "prefix_state" else "prefix_" ^ role) :: y.Tn.label)
          ~unpadded_dims:(lazy [| 1 |])
          ~padding:(lazy None)
          ()
      in
      Tn.update_memory_mode tn Tn.Virtual provenance;
      Tn.Weak_table.replace owners tn y.Tn.uid;
      tn)

(* {1 The index-only domain}

   A mask is proven a triangle by evaluating it symbolically over the reduction's two coordinates:
   the reduced [r] and the output [q]. A value is either an exact affine form [c_r r + c_q q + c],
   or a function of the difference [d = r - q] alone. Comparisons take the difference of their
   affine operands first, so [range[q] < range[r]] is a function of [d]; any other operation is
   evaluated pointwise through [Ops.interpret_*] once its operands are functions of [d]. A symbol
   other than [r] and [q], a tensor the domain cannot follow, or an affine form that is not a
   function of [d] where one is needed makes the mask unrecognized. A function of [d] is then a
   finite table: the check against the four triangles costs O(extent). *)

type value = Lin of { cr : int; cq : int; c : float } | D of (int -> float)

type domain = {
  r : Idx.symbol;
  q : Idx.symbol;
  lo : int;  (** The loops' common range... *)
  hi : int;  (** ...inclusive. *)
  mutable precs : Ops.prec list;  (** Every precision a mask value was computed or stored at. *)
  mutable values : value list;  (** Every intermediate value, checked representable at the end. *)
}

let lin_of_index dom (i : Idx.axis_index) =
  let* terms, offset = terms_of i in
  List.fold terms
    ~init:(Some (0, 0))
    ~f:(fun acc (coef, s) ->
      let* cr, cq = acc in
      if Idx.equal_symbol s dom.r then Some (cr + coef, cq)
      else if Idx.equal_symbol s dom.q then Some (cr, cq + coef)
      else None)
  |> Option.map ~f:(fun (cr, cq) -> Lin { cr; cq; c = Float.of_int offset })

let to_d = function
  | D f -> Some f
  | Lin { cr; cq; c } when cr + cq = 0 -> Some (fun d -> Float.of_int (cr * d) +. c)
  | Lin _ -> None

(* The exact-integer reach of a precision: every integer of at most this magnitude is representable,
   so index arithmetic within it computes exactly; and whether it holds infinities (a log-mask's
   rejected cells). The signed index precisions are what a selector annotated at index precision
   computes in. Other precisions decline. *)
let exact_bound = function
  | Ops.Double_prec _ -> Some (9007199254740992., true)
  | Ops.Single_prec _ -> Some (16777216., true)
  | Ops.Half_prec _ -> Some (2048., true)
  | Ops.Bfloat16_prec _ -> Some (256., true)
  | Ops.Int32_prec _ -> Some (2147483647., false)
  | Ops.Int64_prec _ -> Some (9007199254740992., false)
  | _ -> None

let representable dom =
  let* bound, infinities =
    List.fold dom.precs
      ~init:(Some (Float.infinity, true))
      ~f:(fun acc p ->
        let* acc, inf = acc in
        let* b, inf' = exact_bound p in
        Some (Float.min acc b, inf && inf'))
  in
  let reach = Float.of_int (Int.max (Int.abs dom.lo) (Int.abs dom.hi)) in
  let fits v = (Float.is_inf v && infinities) || (Float.is_integer v && Float.(abs v <= bound)) in
  Option.some_if
    (List.for_all dom.values ~f:(function
      | Lin { cr = 0; cq = 0; c } -> fits c
      | Lin { cr; cq; c } ->
          Float.is_integer c
          && Float.(
               (of_int (Int.abs cr) *. reach) + (of_int (Int.abs cq) *. reach) + abs c <= bound)
      | D f ->
          List.for_all
            (List.range ~stop:`inclusive (dom.lo - dom.hi) (dom.hi - dom.lo))
            ~f:(fun d -> fits (f d))))
    ()

let max_depth = 8

(* A read of a host constant (an [ndarray] literal, gh-ocannl-1858) at subscripts affine in [r] and
   [q]: the cells it reads over the whole [(r, q)] square, each subscript in range, must be a
   function of [d = r - q] alone -- bit for bit, so a [-0.] and a [0.] differ. O(extent^2) once, the
   size of the data itself; a square past [max_host_square] cells declines. The data's precision and
   the node's join the precisions the values must be exact in. A node without that declaration
   ([Operation.init]: mutable host data by contract) is never read here. *)
let max_host_square = 1 lsl 24

let host_constant_read dom (tn : Tn.t) (idcs : Idx.axis_index array) : value option =
  let* data = Host_inits.find tn in
  let nd = Lazy.force data in
  let dims = Ndarray.dims nd in
  let extent = dom.hi - dom.lo + 1 in
  let* forms =
    Array.to_list idcs
    |> List.map ~f:(fun i ->
        match lin_of_index dom i with Some (Lin { cr; cq; c }) -> Some (cr, cq, c) | _ -> None)
    |> Option.all
  in
  let forms = Array.of_list forms in
  if Array.length forms <> Array.length dims || extent * extent > max_host_square then None
  else
    let span = dom.hi - dom.lo in
    let table = Array.create ~len:((2 * span) + 1) None in
    let cell r q =
      let idx = Array.map forms ~f:(fun (cr, cq, c) -> (cr * r) + (cq * q) + Float.to_int c) in
      if Array.for_alli idx ~f:(fun a k -> k >= 0 && k < dims.(a)) then
        Some (Ndarray.get_as_float nd idx)
      else None
    in
    let consistent =
      List.for_all (List.range ~stop:`inclusive dom.lo dom.hi) ~f:(fun r ->
          List.for_all (List.range ~stop:`inclusive dom.lo dom.hi) ~f:(fun q ->
              match cell r q with
              | None -> false
              | Some v -> (
                  let slot = r - q + span in
                  match table.(slot) with
                  | None ->
                      table.(slot) <- Some v;
                      true
                  | Some w -> Int64.equal (Int64.bits_of_float v) (Int64.bits_of_float w))))
    in
    if not consistent then None
    else (
      dom.precs <- Ndarray.get_prec nd :: Lazy.force tn.Tn.storage_prec :: dom.precs;
      let table = Array.map table ~f:(fun v -> Option.value_exn v) in
      Some (D (fun d -> table.(d + span))))

(* The affine form's coefficients stay far inside an OCaml [int], so its arithmetic cannot wrap: a
   scale is an integer of at most f32's exact reach, and a coefficient past [2^30] declines (no
   index grid representable in any supported precision needs one). *)
let small_int c = Float.is_integer c && Float.(abs c <= 16777216.)
let coefficient_bound = 1 lsl 30

(* [eval r dom ~before ~depth sub s]: the value of [s], read at a statement at position [before]
   whose loop symbols [sub] maps to indices over [r] and [q]. A read follows the read node's one
   definition -- a single-writer elementwise nest covering its whole node, at a position before the
   read, unpadded, read at plain subscripts -- and the definition's body computes at its node's
   precision, which joins the precisions the values must be exact in. *)
let rec eval (rt : routine) dom ~before ~depth sub (s : LL.scalar_t) : value option =
  let note v =
    match v with
    | Lin { cr; cq; _ } when Int.abs cr > coefficient_bound || Int.abs cq > coefficient_bound ->
        None
    | _ ->
        dom.values <- v :: dom.values;
        Some v
  in
  let arg (s, prec) =
    dom.precs <- prec :: dom.precs;
    eval rt dom ~before ~depth sub s
  in
  let pointwise_d op vs = Option.all (List.map vs ~f:to_d) |> Option.map ~f:(fun fs -> D (op fs)) in
  match s with
  | LL.Constant c -> note (Lin { cr = 0; cq = 0; c })
  | LL.Embed_index i ->
      let* i = subst_idx sub i in
      let* v = lin_of_index dom i in
      note v
  | LL.Get (tn, idcs) -> (
      if depth <= 0 || Option.is_some (Lazy.force tn.Tn.padding) then None
      else
        let* idcs =
          Array.to_list idcs
          |> List.map ~f:(subst_idx sub)
          |> Option.all |> Option.map ~f:Array.of_list
        in
        match writers rt tn with
        | [ pos ] when pos < before -> (
            match rt.nests.(pos) with
            | _ when match rt.stmts.(pos) with LL.Zero_out _ -> true | _ -> false ->
                (* A zero fill: the lowering of a [0.] literal. *)
                dom.precs <- Lazy.force tn.Tn.storage_prec :: dom.precs;
                note (Lin { cr = 0; cq = 0; c = 0. })
            | Some n when pointwise n && covers_whole n && Array.length n.idcs = Array.length idcs
              ->
                let pairs = Array.zip_exn n.idcs idcs |> Array.to_list in
                let* sub' =
                  List.fold pairs ~init:(Some []) ~f:(fun acc pair ->
                      let* acc = acc in
                      match pair with
                      | Idx.Iterator s, ((Idx.Iterator _ | Idx.Fixed_idx _) as i) ->
                          Some ((s, i) :: acc)
                      | Idx.Fixed_idx k, Idx.Fixed_idx k' when k = k' -> Some acc
                      | _ -> None)
                in
                let sub' s = List.Assoc.find sub' ~equal:Idx.equal_symbol s in
                dom.precs <- Lazy.force tn.Tn.storage_prec :: dom.precs;
                let* v = eval rt dom ~before:pos ~depth:(depth - 1) sub' n.llsc in
                note v
            | _ -> None)
        | [] when Tn.known_host_constant tn ->
            (* No writer here, and values declared forever equal to the registered host data, which
               no routine or host write can change (gh-ocannl-2021): the data is the mask. *)
            let* v = host_constant_read dom tn idcs in
            note v
        | _ -> None)
  | LL.Unop (op, a) -> (
      let* a = arg a in
      match (op, a) with
      | Ops.Neg, Lin { cr; cq; c } -> note (Lin { cr = -cr; cq = -cq; c = -.c })
      (* Pointwise evaluation is in double, which has no rounding to a narrow precision. *)
      | Ops.Round_prec, _ -> None
      | _ ->
          let* v = pointwise_d (fun fs d -> Ops.interpret_unop op ((List.hd_exn fs) d)) [ a ] in
          note v)
  | LL.Binop (op, a, b) -> (
      let* a = arg a in
      let* b = arg b in
      let binary fs d =
        match fs with [ f; g ] -> Ops.interpret_binop op (f d) (g d) | _ -> assert false
      in
      match (op, a, b) with
      | (Ops.Add | Ops.Sub), Lin x, Lin y ->
          let sg = match op with Ops.Sub -> -1 | _ -> 1 in
          note
            (Lin
               {
                 cr = x.cr + (sg * y.cr);
                 cq = x.cq + (sg * y.cq);
                 c = x.c +. (Float.of_int sg *. y.c);
               })
      | Ops.Mul, Lin x, Lin y
        when (y.cr = 0 && y.cq = 0 && small_int y.c) || (x.cr = 0 && x.cq = 0 && small_int x.c) ->
          (* Scaling by an integer constant. *)
          let cr, cq, c, k =
            if y.cr = 0 && y.cq = 0 && small_int y.c then (x.cr, x.cq, x.c, y.c)
            else (y.cr, y.cq, y.c, x.c)
          in
          let ki = Float.to_int k in
          note (Lin { cr = ki * cr; cq = ki * cq; c = k *. c })
      | (Ops.Cmplt | Ops.Cmple | Ops.Cmpeq | Ops.Cmpne), Lin x, Lin y
        when Float.is_finite x.c && Float.is_finite y.c ->
          (* Exact integers compare as their difference compares with zero; infinities do not ([inf
             <= inf], but not [inf - inf <= 0]), and compare directly below. *)
          let diff = Lin { cr = x.cr - y.cr; cq = x.cq - y.cq; c = x.c -. y.c } in
          let* f = to_d diff in
          note (D (fun d -> Ops.interpret_binop op (f d) 0.))
      | _ ->
          let* v = pointwise_d binary [ a; b ] in
          note v)
  | LL.Ternop (op, a, b, c) ->
      let* a = arg a in
      let* b = arg b in
      let* c = arg c in
      let* v =
        pointwise_d
          (fun fs d ->
            match fs with
            | [ f; g; h ] -> Ops.interpret_ternop op (f d) (g d) (h d)
            | _ -> assert false)
          [ a; b; c ]
      in
      note v
  | LL.Constant_bits _ | LL.Local_scope _ | LL.Get_local _ | LL.Get_dynamic _
  | LL.Get_merge_buffer _ ->
      None

(* A mask factor: index-only, a function of [d], and not constant over the range (a constant factor
   is payload). *)
let mask_of rt dom ~before s =
  let saved_precs = dom.precs and saved_values = dom.values in
  let restore () =
    dom.precs <- saved_precs;
    dom.values <- saved_values
  in
  match Option.bind (eval rt dom ~before ~depth:max_depth (fun _ -> None) s) ~f:to_d with
  | Some f ->
      let span = List.range ~stop:`inclusive (dom.lo - dom.hi) (dom.hi - dom.lo) in
      let v0 = f (List.hd_exn span) in
      if List.exists span ~f:(fun d -> not (Float.equal (f d) v0)) then Some f
      else (
        restore ();
        None)
  | None ->
      restore ();
      None

(* {1 Matching} *)

type triangle = { direction : LL.scan_direction; inclusive : bool }

(* Which triangle a selection over [d = r - q] is, if any: [r <= q] is a prefix ending at [q], [r >=
   q] a suffix starting at it. *)
let triangle dom (sel : int -> bool) =
  let span = List.range ~stop:`inclusive (dom.lo - dom.hi) (dom.hi - dom.lo) in
  let is p = List.for_all span ~f:(fun d -> Bool.equal (sel d) (p d)) in
  List.find_map
    [
      ((fun d -> d <= 0), { direction = LL.Forward; inclusive = true });
      ((fun d -> d < 0), { direction = LL.Forward; inclusive = false });
      ((fun d -> d >= 0), { direction = LL.Backward; inclusive = true });
      ((fun d -> d > 0), { direction = LL.Backward; inclusive = false });
    ]
    ~f:(fun (p, t) -> Option.some_if (is p) t)

let neutral = function
  | Ops.Add -> Some 0.
  | Ops.Max -> Some Float.neg_infinity
  | Ops.Min -> Some Float.infinity
  | _ -> None

(* [reduction] with [Min] among the commutative operators. *)
let accumulation (n : nest) =
  match reduction n with
  | Some _ as found -> found
  | None -> (
      match n.llsc with
      | LL.Binop (Ops.Min, (rhs, _), (LL.Get (tn, idcs), _))
        when Tn.equal tn n.tn
             && [%equal: Idx.axis_index array] idcs n.idcs
             && not (LL.scalar_mentions_tn n.tn rhs) ->
          Some (Ops.Min, rhs)
      | _ -> None)

let rec factors = function
  | LL.Binop (Ops.Mul, (a, _), (b, _)) -> factors a @ factors b
  | s -> [ s ]

(* Whether [s], read at [before], is the constant [v]: a literal, or a read of a node the domain
   follows to one (a scalar literal of [%op] is a node with a constant fill). *)
let is_const rt dom ~before v s =
  match s with
  | LL.Constant c -> Float.equal c v
  | _ -> (
      (* Held to the masks' standard: a constant derived through stored definitions is the value the
         composed form reads only when every intermediate is exact where it was computed. *)
      let scratch = { dom with precs = []; values = [] } in
      match eval rt scratch ~before ~depth:max_depth (fun _ -> None) s with
      | Some (Lin { cr = 0; cq = 0; c }) ->
          Float.equal c v && Option.is_some (representable scratch)
      | _ -> false)

let rec index_mentions sym (s : LL.scalar_t) =
  let idx i =
    match terms_of i with
    | Some (terms, _) -> List.exists terms ~f:(fun (_, s) -> Idx.equal_symbol s sym)
    | None -> true
  in
  match s with
  | LL.Get (_, idcs) -> Array.exists idcs ~f:idx
  | LL.Embed_index i -> idx i
  | LL.Constant _ | LL.Constant_bits _ -> false
  | LL.Unop (_, (a, _)) -> index_mentions sym a
  | LL.Binop (_, (a, _), (b, _)) -> index_mentions sym a || index_mentions sym b
  | LL.Ternop (_, (a, _), (b, _), (c, _)) ->
      index_mentions sym a || index_mentions sym b || index_mentions sym c
  | LL.Local_scope _ | LL.Get_local _ | LL.Get_dynamic _ | LL.Get_merge_buffer _ -> true

let rec read_nodes (s : LL.scalar_t) =
  match s with
  | LL.Get (tn, _) -> [ tn ]
  | LL.Embed_index _ | LL.Constant _ | LL.Constant_bits _ -> []
  | LL.Unop (_, (a, _)) -> read_nodes a
  | LL.Binop (_, (a, _), (b, _)) -> read_nodes a @ read_nodes b
  | LL.Ternop (_, (a, _), (b, _), (c, _)) -> read_nodes a @ read_nodes b @ read_nodes c
  | LL.Local_scope _ | LL.Get_local _ | LL.Get_dynamic _ | LL.Get_merge_buffer _ -> []

(* The reduction's operand, with a read of a materialized node replaced by the node's definition
   while the operand is one read: the masked operand of a max ([where tri a (-inf)] stored, then
   reduced) is only visible through its definition. Only through nodes at the target's precision --
   the definition then computes where the reduction does, at the same width -- each definition ahead
   of the read it stands for, and only while the nodes the last one reads have no writer between it
   and the reduction. *)
let operand rt ~(y : Tn.t) ~pos rhs =
  let prec = Lazy.force y.Tn.storage_prec in
  let rec go depth first s =
    match s with
    | LL.Get (tn, idcs) when depth > 0 && Ops.equal_prec (Lazy.force tn.Tn.storage_prec) prec -> (
        match definition rt tn with
        | Some (dpos, n)
          when dpos < first (* ahead of the read it replaces, not only of the reduction *)
               && covers_whole n
               && Option.is_none (Lazy.force tn.Tn.padding)
               && Array.length n.idcs = Array.length idcs -> (
            let pairs = Array.zip_exn n.idcs idcs |> Array.to_list in
            let plain =
              List.for_all pairs ~f:(function
                | Idx.Iterator _, (Idx.Iterator _ | Idx.Fixed_idx _) -> true
                | Idx.Fixed_idx k, Idx.Fixed_idx k' -> k = k'
                | _ -> false)
            in
            let sub =
              List.filter_map pairs ~f:(function Idx.Iterator s, i -> Some (s, i) | _ -> None)
            in
            let idx = function
              | Idx.Iterator s -> List.Assoc.find sub ~equal:Idx.equal_symbol s
              | Idx.Fixed_idx _ as i -> Some i
              | _ -> None
            in
            if not plain then (s, first)
            else
              match map_scalar ~idx ~get:(fun _ _ -> None) n.llsc with
              | Some (body, _) -> go (depth - 1) (Int.min first dpos) body
              | None -> (s, first))
        | _ -> (s, first))
    | _ -> (s, first)
  in
  let s, first = go 4 pos rhs in
  let moved_ok =
    first = pos
    || List.for_all (read_nodes s) ~f:(fun tn ->
        (not (Tn.equal tn y)) && List.for_all (writers rt tn) ~f:(fun w -> w < first || w >= pos))
  in
  Option.some_if moved_ok (s, first)

type site = {
  rows : loop list;
  y : Tn.t;
  y_idcs : Idx.axis_index array;
  op : Ops.binop;
  scan : loop;  (** The output coordinate's loop, which the scan runs over. *)
  shape : triangle;
  payload : LL.scalar_t;  (** In the scan's symbols. *)
}

(* Split the operand into a payload and a triangle, for one choice of [r] and [q]. The mask's reads
   ran at [pos] in the original: the reduction, or the earliest definition the operand unfolded. *)
let split rt dom ~pos ~op (s : LL.scalar_t) : (LL.scalar_t * triangle) option =
  let* fill = neutral op in
  let of_mask (f : int -> float) ~valid ~sel =
    let span = List.range ~stop:`inclusive (dom.lo - dom.hi) (dom.hi - dom.lo) in
    if List.for_all span ~f:(fun d -> valid (f d)) then triangle dom (fun d -> sel (f d)) else None
  in
  let truthy v = not (Float.equal v 0.) in
  let selected =
    match s with
    | LL.Ternop (Ops.Where, (c, _), (a, _), (b, _)) when is_const rt dom ~before:pos fill b ->
        let* f = mask_of rt dom ~before:pos c in
        let* t = of_mask f ~valid:(fun _ -> true) ~sel:truthy in
        Some (a, t)
    | LL.Ternop (Ops.Where, (c, _), (a, _), (b, _)) when is_const rt dom ~before:pos fill a ->
        let* f = mask_of rt dom ~before:pos c in
        let* t = of_mask f ~valid:(fun _ -> true) ~sel:(fun v -> not (truthy v)) in
        Some (b, t)
    | LL.Binop (Ops.Add, (a, _), (b, _)) when not (Ops.equal_binop op Ops.Add) -> (
        (* The tropical form: a log-mask of [0] and the neutral element. *)
        let log_mask m v =
          let* f = mask_of rt dom ~before:pos m in
          let* t =
            of_mask f
              ~valid:(fun x -> Float.equal x 0. || Float.equal x fill)
              ~sel:(fun x -> Float.equal x 0.)
          in
          Some (v, t)
        in
        match log_mask b a with Some _ as found -> found | None -> log_mask a b)
    | _ when Ops.equal_binop op Ops.Add -> (
        let masks, payload =
          List.partition_map (factors s) ~f:(fun x ->
              match mask_of rt dom ~before:pos x with Some f -> First f | None -> Second x)
        in
        match masks with
        | [] -> None
        | _ ->
            let f d = List.fold masks ~init:1. ~f:(fun acc g -> acc *. g d) in
            let* t = of_mask f ~valid:(fun x -> Float.equal x 0. || Float.equal x 1.) ~sel:truthy in
            let v =
              match payload with
              | [] -> LL.Constant 1.
              | x :: xs ->
                  List.fold xs ~init:x ~f:(fun acc x ->
                      LL.apply_op (Ops.Binop Ops.Mul) [| acc; x |])
            in
            Some (v, t))
    | _ -> None
  in
  let* v, t = selected in
  let* () = representable dom in
  Some (v, t)

let find_site rt pos (n : nest) : site option =
  let* op, rhs = accumulation n in
  let* _ = neutral op in
  (* Code the census cannot see anywhere ahead of the reduction -- staged code, say -- may write the
     selector or the payload's sources where [writers] does not look, so nothing is proven across
     it: neither a mask's definitions nor an unfolded operand's reads. *)
  let* () = Option.some_if (not (Array.existsi rt.opaque ~f:(fun i o -> o && i <= pos))) () in
  let y = n.tn in
  let* _ = state_prec y in
  let* outputs =
    Array.to_list n.idcs
    |> List.map ~f:(function
      | Idx.Iterator s -> Some (Some s)
      | Idx.Fixed_idx _ -> Some None
      | _ -> None)
    |> Option.all |> Option.map ~f:List.filter_opt
  in
  let* () = Option.some_if (not (List.contains_dup outputs ~compare:Idx.compare_symbol)) () in
  let is_output s = List.mem outputs s ~equal:Idx.equal_symbol in
  let* r =
    match List.filter n.loops ~f:(fun lp -> not (is_output lp.index)) with
    | [ r ] when r.from_ <= r.to_ -> Some r
    | _ -> None
  in
  let* s, first = operand rt ~y ~pos rhs in
  let* () = Option.some_if (not (LL.scalar_mentions_tn y s)) () in
  List.find_map n.loops ~f:(fun q ->
      if (not (is_output q.index)) || q.from_ <> r.from_ || q.to_ <> r.to_ then None
      else
        let dom =
          {
            r = r.index;
            q = q.index;
            lo = r.from_;
            hi = r.to_;
            precs = [ Lazy.force y.Tn.storage_prec ];
            values = [];
          }
        in
        let* v, shape = split rt dom ~pos:first ~op s in
        let* () = Option.some_if (not (index_mentions q.index v)) () in
        let* payload =
          subst_scalar
            (fun s -> Option.some_if (Idx.equal_symbol s r.index) (Idx.Iterator q.index))
            v
        in
        let rows =
          List.filter n.loops ~f:(fun lp ->
              not (Idx.equal_symbol lp.index r.index || Idx.equal_symbol lp.index q.index))
        in
        Some { rows; y; y_idcs = n.idcs; op; scan = q; shape; payload })

(* {1 The extremum's gradient gate (gh-ocannl-1858)}

   The backward of a cumulative max (min) through [einmax1]'s product-space gate lowers to

   - [P[i, o] = where(M(i, o), v[i], -inf)], [Y[o] = -inf], [Y[o] = max(Y[o], P[i, o])] -- the
   forward, which [find_site] reads; - [C[o, i] = (Y[o] = P[i, o])], then [A[i, o] = 0], [A[i, o] +=
   where(C[o, i], g[o], 0)]; - [T[i] += where(M(i, o), A[i, o], 0)], reducing over [o]: O(n^2).

   So [T[i] += sum over the [o] selecting [i] of [Y[o] == v[i]] g[o]]: every position attaining an
   output's extremum takes its whole upstream gradient (duplication, the tie rule [einmax1]
   documents). [Y] reduces over nested selections, so it is monotone along the forward's direction,
   and with [o*] the first output selecting [i], [Y[o] == v[i]] iff [v[i] == Y[o*]] and [Y[o] ==
   Y[o*]] -- a sandwich, in the IEEE order where the two zeros are equal. The outputs equal to
   [Y[o*]] from [o*] on are a run, so [T[i] += [v[i] == Y[o*]] R[o*]], where [R] sums [g] over the
   rest of a run: a scan in the reverse direction carrying the run's sum and the previous [Y],
   restarting the sum where [Y] changes. The sum is regrouped (from the run's end), the comparisons
   are the composed form's own. *)

type gate = {
  g_rows : loop list;
  g_t : Tn.t;
  g_t_idcs : Idx.axis_index array;
  g_scan : loop;  (** [T]'s output loop over [i]: the scan's position [p]. *)
  g_shape : triangle;  (** [T]'s own triangle: the reverse of the forward's. *)
  g_y : LL.scalar_t;  (** [Y] at the forward output [p]. *)
  g_y_prec : Ops.prec;
  g_v : LL.scalar_t;  (** The forward payload at [p]. *)
  g_eq : LL.scalar_t -> LL.scalar_t -> LL.scalar_t;
      (** [v == y] with [C]'s own operand order and precisions. *)
  g_u : LL.scalar_t;  (** The upstream gradient at the forward output [p]. *)
  g_u_prec : Ops.prec;
}

let flip = function LL.Forward -> LL.Backward | LL.Backward -> LL.Forward

(* [sub] from a nest's target subscripts (plain iterators, or a fixed index matched exactly) to the
   indices a read of its node uses. *)
let unify (target : Idx.axis_index array) (read : Idx.axis_index array) =
  if Array.length target <> Array.length read then None
  else
    let* pairs =
      Array.to_list (Array.zip_exn target read)
      |> List.map ~f:(function
        | Idx.Iterator s, i -> Some (Some (s, i))
        | Idx.Fixed_idx k, Idx.Fixed_idx k' when k = k' -> Some None
        | _ -> None)
      |> Option.all
    in
    let pairs = List.filter_opt pairs in
    Some (fun s -> List.Assoc.find pairs ~equal:Idx.equal_symbol s)

let instantiate (n : nest) read =
  Option.bind (unify n.idcs read) ~f:(fun sub -> subst_scalar sub n.llsc)

let fill_of rt pos ~(tn : Tn.t) =
  match rt.stmts.(pos) with
  | LL.Zero_out tn' when Tn.equal tn tn' -> Some 0.
  | _ -> (
      match rt.nests.(pos) with
      | Some ({ llsc = LL.Constant c; _ } as n) when Tn.equal n.tn tn && covers_whole n -> Some c
      | _ -> None)

let find_gate rt pos (n : nest) : gate option =
  let* op, rhs = reduction n in
  let* () = Option.some_if (Ops.equal_binop op Ops.Add) () in
  let* () = Option.some_if (not (Array.existsi rt.opaque ~f:(fun i o -> o && i <= pos))) () in
  let t = n.tn in
  let t_prec = Lazy.force t.Tn.storage_prec in
  let* _ = state_prec t in
  let* outputs =
    Array.to_list n.idcs
    |> List.map ~f:(function
      | Idx.Iterator s -> Some (Some s)
      | Idx.Fixed_idx _ -> Some None
      | _ -> None)
    |> Option.all |> Option.map ~f:List.filter_opt
  in
  let* () = Option.some_if (not (List.contains_dup outputs ~compare:Idx.compare_symbol)) () in
  let is_output s = List.mem outputs s ~equal:Idx.equal_symbol in
  let* r =
    match List.filter n.loops ~f:(fun lp -> not (is_output lp.index)) with
    | [ r ] when r.from_ < r.to_ -> Some r
    | _ -> None
  in
  List.find_map n.loops ~f:(fun q ->
      if (not (is_output q.index)) || q.from_ <> r.from_ || q.to_ <> r.to_ then None
      else
        let dom =
          { r = r.index; q = q.index; lo = r.from_; hi = r.to_; precs = [ t_prec ]; values = [] }
        in
        let at_p s =
          subst_scalar
            (fun s -> Option.some_if (Idx.equal_symbol s r.index) (Idx.Iterator q.index))
            s
        in
        let* a, shape = split rt dom ~pos ~op:Ops.Add rhs in
        (* [A[i, o]]: one zero fill, then one accumulation covering it, both ahead of [T]'s. *)
        let* a_tn, a_idcs = match a with LL.Get (tn, idcs) -> Some (tn, idcs) | _ -> None in
        let* () = Option.some_if (index_mentions q.index a) () in
        let a_prec = Lazy.force a_tn.Tn.storage_prec in
        let* () = Option.some_if (Ops.equal_prec a_prec t_prec) () in
        let* z_pos, g_pos =
          match writers rt a_tn with [ z; g ] when g < pos -> Some (z, g) | _ -> None
        in
        let* zero = fill_of rt z_pos ~tn:a_tn in
        let* () = Option.some_if (Float.equal zero 0.) () in
        let* g_nest = rt.nests.(g_pos) in
        let* () = Option.some_if (covers_whole g_nest) () in
        let* g_op, g_rhs = reduction g_nest in
        let* () = Option.some_if (Ops.equal_binop g_op Ops.Add) () in
        (* [A += where(C, u, 0)], in [T]'s coordinates. *)
        let* g_rhs =
          Option.bind (unify g_nest.idcs a_idcs) ~f:(fun sub -> subst_scalar sub g_rhs)
        in
        let* c_tn, c_idcs, u, u_prec =
          match g_rhs with
          | LL.Ternop (Ops.Where, (LL.Get (c, ci), _), (u, u_prec), (z, _))
            when is_const rt dom ~before:g_pos 0. z ->
              Some (c, ci, u, u_prec)
          | _ -> None
        in
        let* () = Option.some_if (Ops.equal_prec u_prec a_prec) () in
        let* () = Option.some_if (not (index_mentions q.index u)) () in
        let* () = Option.some_if (not (LL.scalar_mentions_tn t u)) () in
        (* [C[o, i] = (Y[o] = P[i, o])], defined once ahead of the accumulation. *)
        let* c_pos, c_nest = definition rt c_tn in
        let* () = Option.some_if (c_pos < g_pos && covers_whole c_nest) () in
        let* c_body = instantiate c_nest c_idcs in
        let* (ya, ia, pa), (yb, ib, pb) =
          match c_body with
          | LL.Binop (Ops.Cmpeq, (LL.Get (ya, ia), pa), (LL.Get (yb, ib), pb)) ->
              Some ((ya, ia, pa), (yb, ib, pb))
          | _ -> None
        in
        let cmp x y = LL.Binop (Ops.Cmpeq, (x, pa), (y, pb)) in
        (* Either operand may be the extremum. *)
        List.find_map
          [ ((ya, ia), (yb, ib), fun ~v ~y -> cmp y v); ((yb, ib), (ya, ia), fun ~v ~y -> cmp v y) ]
          ~f:(fun ((y_tn, y_read), (p_tn, p_read), eq) ->
            (* [Y]: the neutral fill, then a forward the scan recognizer accepts, reading [P]. *)
            let* yf_pos, yr_pos =
              match writers rt y_tn with [ f; r ] when r < c_pos -> Some (f, r) | _ -> None
            in
            (* The equality computes at its node's precision (a homogeneous operation's arguments
               take the enclosing one), so the scan's -- at [T]'s -- is the same comparison only
               when [C], [Y] and [T] share a precision; the run test, at the state's widening of it,
               is then exact too. *)
            let y_prec = Lazy.force y_tn.Tn.storage_prec in
            let* () =
              Option.some_if
                (Ops.equal_prec (Lazy.force c_tn.Tn.storage_prec) y_prec
                && Ops.equal_prec y_prec t_prec)
                ()
            in
            let* y_nest = rt.nests.(yr_pos) in
            let* fwd = find_site rt yr_pos y_nest in
            let* fill = neutral fwd.op in
            let* () =
              Option.some_if (Ops.equal_binop fwd.op Ops.Max || Ops.equal_binop fwd.op Ops.Min) ()
            in
            let* y_fill = fill_of rt yf_pos ~tn:y_tn in
            let* () = Option.some_if (Float.equal y_fill fill) () in
            let* y_op, y_rhs = accumulation y_nest in
            let* () = Option.some_if (Ops.equal_binop y_op fwd.op) () in
            let* p_fwd_read =
              match y_rhs with LL.Get (p', ip) when Tn.equal p' p_tn -> Some ip | _ -> None
            in
            (* The forward's coordinates in [T]'s: its outputs as [C] reads [Y], its reduced loop
               [i]. *)
            let y_outputs =
              Array.to_list y_nest.idcs
              |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
            in
            let* y_r =
              match
                List.filter y_nest.loops ~f:(fun lp ->
                    not (List.mem y_outputs lp.index ~equal:Idx.equal_symbol))
              with
              | [ lp ] -> Some lp
              | _ -> None
            in
            let* sub_y = unify y_nest.idcs y_read in
            let phi s =
              if Idx.equal_symbol s y_r.index then Some (Idx.Iterator q.index) else sub_y s
            in
            let* () =
              Option.some_if
                (match phi fwd.scan.index with
                | Some (Idx.Iterator s) -> Idx.equal_symbol s r.index
                | _ -> false)
                ()
            in
            let* () = Option.some_if (fwd.scan.from_ = r.from_ && fwd.scan.to_ = r.to_) () in
            (* Every other loop of the forward is a row, and stays one: its image reads neither of
               [T]'s triangular coordinates, or the scan would walk across rows (and could leave an
               index unbound). *)
            let triangular sym = Idx.equal_symbol sym r.index || Idx.equal_symbol sym q.index in
            let* () =
              Option.some_if
                (List.for_all y_nest.loops ~f:(fun lp ->
                     Idx.equal_symbol lp.index y_r.index
                     || Idx.equal_symbol lp.index fwd.scan.index
                     ||
                     match Option.bind (phi lp.index) ~f:terms_of with
                     | Some (terms, _) -> not (List.exists terms ~f:(fun (_, s) -> triangular s))
                     | None -> false))
                ()
            in
            let* p_mapped =
              Array.to_list p_fwd_read
              |> List.map ~f:(subst_idx phi)
              |> Option.all |> Option.map ~f:Array.of_list
            in
            let* () = Option.some_if ([%equal: Idx.axis_index array] p_mapped p_read) () in
            (* One relation: [T]'s triangle is the forward's, read from the other side. *)
            let* () =
              Option.some_if
                (LL.equal_scan_direction shape.direction (flip fwd.shape.direction)
                && Bool.equal shape.inclusive fwd.shape.inclusive)
                ()
            in
            (* What the scan reads at [T]'s position: nothing writes it in between. *)
            let* () =
              Option.some_if (not (List.exists [ y_tn; p_tn; a_tn; c_tn ] ~f:(Tn.equal t))) ()
            in
            let at_fwd_p s =
              subst_scalar
                (fun s ->
                  if Idx.equal_symbol s fwd.scan.index then Some (Idx.Iterator q.index) else phi s)
                s
            in
            let* v = at_fwd_p fwd.payload in
            let* () = Option.some_if (not (LL.scalar_mentions_tn t v)) () in
            let quiet ~from tns =
              List.for_all tns ~f:(fun tn ->
                  List.for_all (writers rt tn) ~f:(fun w -> w < from || w >= pos))
            in
            let* () =
              Option.some_if
                (quiet ~from:yr_pos (read_nodes v) && quiet ~from:g_pos (read_nodes u))
                ()
            in
            let* y_at = at_p (LL.Get (y_tn, y_read)) in
            let* u_at = at_p u in
            let* () = representable dom in
            let rows =
              List.filter n.loops ~f:(fun lp ->
                  not (Idx.equal_symbol lp.index r.index || Idx.equal_symbol lp.index q.index))
            in
            Some
              {
                g_rows = rows;
                g_t = t;
                g_t_idcs = n.idcs;
                g_scan = q;
                g_shape = shape;
                g_y = y_at;
                g_y_prec = Lazy.force y_tn.Tn.storage_prec;
                g_v = v;
                g_eq = (fun v y -> eq ~v ~y);
                g_u = u_at;
                g_u_prec = u_prec;
              }))

(* {1 Emission} *)

let emit ~ordinal (s : site) : LL.t =
  let open LL in
  let prec = Option.value_exn (state_prec s.y) in
  let st = state_node ~y:s.y ~ordinal prec in
  let fill = Option.value_exn (neutral s.op) in
  let c = { prev = get_scope st; next = get_scope st; init = Constant fill } in
  let combine a b = apply_op (Ops.Binop s.op) [| a; b |] in
  (* An exclusive scan's last step folds in a payload no output reads -- [v] at the scan's final
     position, which the composed form never selected, so a payload selected by a [Where] may not
     even be in range there. That step folds the neutral element instead, behind an index guard, the
     form lowering's own clamps take, which the renderers honor as a guard. *)
  let payload =
    if s.shape.inclusive then s.payload
    else
      let iprec = Ops.index_prec () in
      let at = Embed_index (Idx.Iterator s.scan.index) in
      let cond =
        match s.shape.direction with
        | Forward -> Binop (Ops.Cmplt, (at, iprec), (Constant (Float.of_int s.scan.to_), iprec))
        | Backward -> Binop (Ops.Cmplt, (Constant (Float.of_int s.scan.from_), iprec), (at, iprec))
      in
      apply_op (Ops.Ternop Ops.Where) [| cond; s.payload; Constant fill |]
  in
  let step = Set_local (c.next, combine (Get_local c.prev) payload) in
  let store x =
    Set { tn = s.y; idcs = s.y_idcs; llsc = combine (Get (s.y, s.y_idcs)) x; debug = "" }
  in
  let body =
    if s.shape.inclusive then unflat_lines [ step; store (Get_local c.next) ]
    else unflat_lines [ store (Get_local c.prev); step ]
  in
  wrap s.rows
    (Scan_loop
       {
         index = s.scan.index;
         from_ = s.scan.from_;
         to_ = s.scan.to_;
         direction = s.shape.direction;
         carried = [ c ];
         body;
       })

let emit_gate ~ordinal (g : gate) : LL.t =
  let open LL in
  let sprec = Option.value_exn (state_prec g.g_t) in
  let yprec = Option.value_exn (state_prec_of_prec g.g_y_prec) in
  let sum_tn = state_node ~y:g.g_t ~ordinal sprec in
  let ext_tn = state_node ~role:"extremum" ~y:g.g_t ~ordinal yprec in
  let s = { prev = get_scope sum_tn; next = get_scope sum_tn; init = Constant 0. } in
  let e = { prev = get_scope ext_tn; next = get_scope ext_tn; init = Constant 0. } in
  let where_ c a b = apply_op (Ops.Ternop Ops.Where) [| c; a; b |] in
  (* The run continues while the extremum holds; the first step's [prev] is [init], under a zero
     sum, so it never matters. *)
  let same = Binop (Ops.Cmpeq, (Get_local e.prev, g.g_y_prec), (g.g_y, g.g_y_prec)) in
  let sum_step =
    Set_local
      ( s.next,
        Binop (Ops.Add, (g.g_u, g.g_u_prec), (where_ same (Get_local s.prev) (Constant 0.), sprec))
      )
  in
  let ext_step = Set_local (e.next, g.g_y) in
  let store x =
    Set
      {
        tn = g.g_t;
        idcs = g.g_t_idcs;
        llsc = apply_op (Ops.Binop Ops.Add) [| Get (g.g_t, g.g_t_idcs); x |];
        debug = "";
      }
  in
  let body =
    if g.g_shape.inclusive then
      unflat_lines
        [ sum_step; ext_step; store (where_ (g.g_eq g.g_v g.g_y) (Get_local s.next) (Constant 0.)) ]
    else
      (* A strict selection's first output is the one after [i]: the state before the step. The
         scan's first position selects nothing, and its payload -- which the composed form never
         compared -- may be out of range there, so that step stays behind an index guard. *)
      let iprec = Ops.index_prec () in
      let at = Embed_index (Idx.Iterator g.g_scan.index) in
      let not_first =
        match g.g_shape.direction with
        | Backward -> Binop (Ops.Cmplt, (at, iprec), (Constant (Float.of_int g.g_scan.to_), iprec))
        | Forward -> Binop (Ops.Cmplt, (Constant (Float.of_int g.g_scan.from_), iprec), (at, iprec))
      in
      unflat_lines
        [
          store
            (where_ not_first
               (where_ (g.g_eq g.g_v (Get_local e.prev)) (Get_local s.prev) (Constant 0.))
               (Constant 0.));
          sum_step;
          ext_step;
        ]
  in
  wrap g.g_rows
    (Scan_loop
       {
         index = g.g_scan.index;
         from_ = g.g_scan.from_;
         to_ = g.g_scan.to_;
         direction = g.g_shape.direction;
         carried = [ s; e ];
         body;
       })

(* The scans into each target already in the routine, by the targets' uids: a later fixpoint round
   numbers its sites after them. *)
let existing_scans (stmts : LL.t array) =
  let counts = Hashtbl.create (module Int) in
  let rec go = function
    | LL.Seq (a, b) ->
        go a;
        go b
    | LL.For_loop { body; _ } -> go body
    | LL.Scan_loop { carried; body; _ } ->
        (* One per scan: the gradient gate's carries two states into one target. *)
        Option.iter
          (List.find_map carried ~f:(fun (c : LL.carried) -> state_owner c.prev.tn))
          ~f:(fun y -> Hashtbl.incr counts y);
        go body
    | _ -> ()
  in
  Array.iter stmts ~f:go;
  counts

let rewrite (llc : LL.t) : LL.t =
  let rt = routine_of llc in
  let sites =
    Array.to_list rt.nests
    |> List.filter_mapi ~f:(fun pos n ->
        Option.bind n ~f:(fun n ->
            if rt.opaque.(pos) then None
            else
              match find_site rt pos n with
              | Some s -> Some (pos, `Prefix s)
              | None -> Option.map (find_gate rt pos n) ~f:(fun g -> (pos, `Gate g))))
  in
  if List.is_empty sites then llc
  else
    let stmts = Array.copy rt.stmts in
    let ordinals = existing_scans rt.stmts in
    List.iter sites ~f:(fun (pos, site) ->
        let target = match site with `Prefix s -> s.y | `Gate g -> g.g_t in
        let ordinal = Option.value (Hashtbl.find ordinals target.Tn.uid) ~default:0 in
        Hashtbl.set ordinals ~key:target.Tn.uid ~data:(ordinal + 1);
        stmts.(pos) <-
          (match site with `Prefix s -> emit ~ordinal s | `Gate g -> emit_gate ~ordinal g));
    LL.unflat_lines (Array.to_list stmts)
