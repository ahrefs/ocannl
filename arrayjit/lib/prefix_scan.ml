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
let state_prec (tn : Tn.t) =
  match Lazy.force tn.Tn.storage_prec with
  | Ops.Double_prec _ as p -> Some p
  | Ops.Single_prec _ | Ops.Half_prec _ | Ops.Bfloat16_prec _ | Ops.Fp8_prec _ -> Some Ops.single
  | _ -> None

let state_node ~(y : Tn.t) ~ordinal prec =
  Hashtbl.find_or_add
    (Tn.memo_bucket minted y ~create:(fun () -> Hashtbl.create (module Site_key)))
    (y.Tn.uid, ordinal, Ops.prec_string prec)
    ~default:(fun () ->
      let tn =
        Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ())
          ~label:("prefix_state" :: y.Tn.label)
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
        List.iter carried ~f:(fun (c : LL.carried) ->
            Option.iter (state_owner c.prev.tn) ~f:(fun y -> Hashtbl.incr counts y));
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
            if rt.opaque.(pos) then None else Option.map (find_site rt pos n) ~f:(fun s -> (pos, s))))
  in
  if List.is_empty sites then llc
  else
    let stmts = Array.copy rt.stmts in
    let ordinals = existing_scans rt.stmts in
    List.iter sites ~f:(fun (pos, s) ->
        let ordinal = Option.value (Hashtbl.find ordinals s.y.Tn.uid) ~default:0 in
        Hashtbl.set ordinals ~key:s.y.Tn.uid ~data:(ordinal + 1);
        stmts.(pos) <- emit ~ordinal s);
    LL.unflat_lines (Array.to_list stmts)
