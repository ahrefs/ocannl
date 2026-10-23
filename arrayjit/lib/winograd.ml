(* The Winograd F(2x2, 3x3) convolution rewrite (gh-ocannl-505); the contract is in the
   interface. *)

open Base
module LL = Low_level
module Tn = Tnode
module Idx = Indexing

let override : bool option ref = ref None
let set_enabled b = override := b

let enabled () =
  match !override with
  | Some b -> b
  | None -> Utils.get_global_flag ~default:false ~arg_name:"winograd_conv"

(* {1 The transform matrices}

   Lavin and Gray's F(2, 3), in the correlation orientation OCANNL's convolutions lower to ([y.(o) =
   sum_k x.(o + k) * w.(k)]): [y = a_t ((g w) * (b_t x))] elementwise in the middle. Every
   coefficient is dyadic, so the kernel transform scales exactly. *)

let b_t =
  [| [| 1.; 0.; -1.; 0. |]; [| 0.; 1.; 1.; 0. |]; [| 0.; -1.; 1.; 0. |]; [| 0.; 1.; 0.; -1. |] |]

let g = [| [| 1.; 0.; 0. |]; [| 0.5; 0.5; 0.5 |]; [| 0.5; -0.5; 0.5 |]; [| 0.; 0.; 1. |] |]
let a_t = [| [| 1.; 1.; 1.; 0. |]; [| 0.; 1.; -1.; -1. |] |]

(* {1 Minted nodes}

   The transformed kernel [U], the transformed input tiles [V] and their per-position products [M]
   are private to the rewrite and live in a namespace of their own. Each is declared
   [Never_virtual]: inlined, [V] would be re-transformed once per output channel and [M] once per
   inverse-transform term. Whether a node is routine scratch or a buffer is the placements' call. *)

(* Its own namespace, apart from [Online_softmax]'s ["rewrite"]: each module numbers its nodes from
   0, and [Tnode.find_namespaced] and the name-only identifiers key on the (namespace, id) pair. *)
let namespace = "winograd"
let provenance = Tn.Site "505:winograd-transform"

let fresh_id =
  let c = ref (-1) in
  fun () ->
    Int.incr c;
    !c

(* Sibling lowerings of one program (placement arms, autotune candidates) must mint the SAME nodes,
   or the analysis cache, which keys nodes by identity, misses on every one of them. The key is the
   whole site -- the matched statement's nodes and index maps, its loop domains -- plus the role, so
   two statements that differ in any of those never share a buffer; the nest's loop symbols, which
   every lowering mints afresh, enter it by their position in the nest (alpha-normalized). Cleared
   with the other node-retaining caches: ahead of an accessibility snapshot, and at a session reset
   through the tier's [Rewrites.reset]. *)
let minted : (string, Tn.t) Hashtbl.t = Hashtbl.create (module String)
let reset () = Hashtbl.clear minted
let () = Tn.before_accessibility_snapshot := reset :: !Tn.before_accessibility_snapshot

(* The identities this module minted, by role: the membership is the marker, which no public field
   of a node can forge. Never cleared -- a node's uid is never reused. *)
let roles : (int, [ `Kernel | `Input | `Product ]) Hashtbl.t = Hashtbl.create (module Int)
let minted_role (tn : Tn.t) = Hashtbl.find roles tn.Tn.uid

let mint ~site_key ~role ~(like : Tn.t) prec dims =
  let role_label =
    match role with `Kernel -> "wino_u" | `Input -> "wino_v" | `Product -> "wino_m"
  in
  Hashtbl.find_or_add minted
    (role_label ^ "|" ^ site_key)
    ~default:(fun () ->
      let tn =
        Tn.create ~namespace (Tn.Specified prec) ~id:(fresh_id ())
          ~label:(role_label :: like.Tn.label)
          ~unpadded_dims:(lazy dims)
          ~padding:(lazy None)
          ()
      in
      Tn.update_memory_mode tn Tn.Never_virtual provenance;
      Hashtbl.set roles ~key:tn.Tn.uid ~data:role;
      tn)

(* {1 Recognition}

   Every [Accum_op] lowers to one top-level nest: serial loops from the outside in and a single
   [Set] at the bottom. The conv's nest is [d[..] := d[..] + a[..] * b[..]] (either operand order,
   or the [FMA] form), where [a] reads the input at two axes [o + k] -- an output iterator plus a
   window iterator -- and [b] reads the kernel at the window iterators. *)

type loop = { index : Idx.symbol; from_ : int; to_ : int }

let nest_of (stmt : LL.t) =
  let rec go loops = function
    | LL.For_loop { index; from_; to_; body; axis = LL.Serial } ->
        go ({ index; from_; to_ } :: loops) body
    | LL.Set { tn; idcs; llsc; _ } -> Some (List.rev loops, tn, idcs, llsc)
    | _ -> None
  in
  go [] stmt

(* One spatial axis of the conv: the output iterator [o] (extent [n_o]), the window iterator [k]
   (which the kernel reads plainly), and the axis of [a] reading [o + k]. *)
type conv_axis = { o : Idx.symbol; n_o : int; k : Idx.symbol; a_axis : int }

type site = {
  loops : loop list;
  d : Tn.t;
  di : Idx.axis_index array;
  a : Tn.t;
  ai : Idx.axis_index array;
  b : Tn.t;
  bi : Idx.axis_index array;
  prec : Ops.prec;
  h : conv_axis;
  w : conv_axis;
  ic : Idx.symbol;
  rebuild : (Idx.axis_index array -> Idx.axis_index array) -> LL.scalar_t;
      (** The statement's right-hand side with every index map passed through the substitution: the
          direct form, for the remainder strips. *)
  key : string;
}

let ( let* ) x f = Option.bind x ~f

(* The product's reads, with how to rebuild the whole right-hand side under an index substitution.
   [rhs] must be [d + x * y] (either order of the addition) or [fma(x, y, d)]. *)
let match_rhs ~d ~di (llsc : LL.scalar_t) =
  let is_d = function
    | LL.Get (tn, idcs), _ -> Tn.equal tn d && Array.equal Idx.equal_axis_index idcs di
    | _ -> false
  in
  let get = function LL.Get (tn, idcs), _ -> Some (tn, idcs) | _ -> None in
  let regen (sc, prec) f =
    match sc with LL.Get (tn, idcs) -> (LL.Get (tn, f idcs), prec) | _ -> assert false
  in
  match llsc with
  | LL.Binop (Ops.Add, dd, (LL.Binop (Ops.Mul, x, y), pm)) when is_d dd ->
      let* x' = get x in
      let* y' = get y in
      Some
        ( x',
          y',
          fun f -> LL.Binop (Ops.Add, regen dd f, (LL.Binop (Ops.Mul, regen x f, regen y f), pm)) )
  | LL.Binop (Ops.Add, (LL.Binop (Ops.Mul, x, y), pm), dd) when is_d dd ->
      let* x' = get x in
      let* y' = get y in
      Some
        ( x',
          y',
          fun f -> LL.Binop (Ops.Add, (LL.Binop (Ops.Mul, regen x f, regen y f), pm), regen dd f) )
  | LL.Ternop (Ops.FMA, x, y, dd) when is_d dd ->
      let* x' = get x in
      let* y' = get y in
      Some (x', y', fun f -> LL.Ternop (Ops.FMA, regen x f, regen y f, regen dd f))
  | _ -> None

let mentions s (idcs : Idx.axis_index array) =
  Array.count idcs ~f:(Idx.axis_index_mentions_symbol s)

let plain_at s (idcs : Idx.axis_index array) =
  Array.find_mapi idcs ~f:(fun i -> function
    | Idx.Iterator s' when Idx.equal_symbol s s' -> Some i
    | _ -> None)

let find_site (stmt : LL.t) : site option =
  let* loops, d, di, llsc = nest_of stmt in
  let* (a0, ai0), (b0, bi0), rebuild = match_rhs ~d ~di llsc in
  let loop_syms = List.map loops ~f:(fun l -> l.index) in
  let is_loop s = List.mem loop_syms s ~equal:Idx.equal_symbol in
  let extent s =
    List.find_map loops ~f:(fun l -> if Idx.equal_symbol l.index s then Some (l.to_ + 1) else None)
  in
  (* The conv components of an index map: axes [o + k] with unit coefficients and offset 0. *)
  let conv_components idcs =
    Array.to_list idcs
    |> List.filter_mapi ~f:(fun i -> function
      | Idx.Affine { symbols = [ (1, s1); (1, s2) ]; offset = 0 } -> Some (i, s1, s2)
      | _ -> None)
  in
  (* [a] is the operand with the two conv components; [b] has none. *)
  let* (a, ai), (b, bi) =
    match (conv_components ai0, conv_components bi0) with
    | [ _; _ ], [] -> Some ((a0, ai0), (b0, bi0))
    | [], [ _; _ ] -> Some ((b0, bi0), (a0, ai0))
    | _ -> None
  in
  let all_plain idcs ~except =
    Array.for_alli idcs ~f:(fun i -> function
      | Idx.Iterator _ | Idx.Fixed_idx _ -> true
      | _ -> List.mem except i ~equal:Int.equal)
  in
  let d_iters =
    Array.to_list di |> List.filter_map ~f:(function Idx.Iterator s -> Some s | _ -> None)
  in
  let is_out s = List.mem d_iters s ~equal:Idx.equal_symbol in
  let axis_of (i, s1, s2) =
    (* Which of the pair is the output iterator: the one [d] reads. *)
    let* o, k =
      match (is_out s1, is_out s2) with
      | true, false -> Some (s1, s2)
      | false, true -> Some (s2, s1)
      | _ -> None
    in
    let* _ = plain_at k bi in
    let* n_o = extent o in
    let* n_k = extent k in
    if n_k = 3 && n_o >= 2 then Some { o; n_o; k; a_axis = i } else None
  in
  let* h, w =
    match conv_components ai with
    | [ c1; c2 ] ->
        let* h = axis_of c1 in
        let* w = axis_of c2 in
        Some (h, w)
    | _ -> None
  in
  let specials = [ h.o; h.k; w.o; w.k ] in
  let is_special s = List.mem specials s ~equal:Idx.equal_symbol in
  (* Separability: each special symbol occurs exactly where the transform assumes and nowhere else
     -- an output iterator once in [d] (plainly) and once in [a] (its conv axis), a window iterator
     once in [a] (its conv axis) and once in [b] (plainly). *)
  let separable =
    List.for_all [ h; w ] ~f:(fun cx ->
        mentions cx.o di = 1
        && Option.is_some (plain_at cx.o di)
        && mentions cx.o ai = 1
        && mentions cx.o bi = 0
        && mentions cx.k di = 0
        && mentions cx.k ai = 1
        && mentions cx.k bi = 1)
    && (not (Idx.equal_symbol h.o w.o))
    && not (Idx.equal_symbol h.k w.k)
  in
  (* The non-special symbols the three maps read, with the role their occurrences give them. *)
  let others idcs =
    Array.to_list idcs
    |> List.filter_map ~f:(function Idx.Iterator s when not (is_special s) -> Some s | _ -> None)
  in
  let d_o = others di and a_o = others ai and b_o = others bi in
  let mem l s = List.mem l s ~equal:Idx.equal_symbol in
  let loops_ok = List.for_all loops ~f:(fun l -> l.from_ = 0) in
  let roles_ok () =
    (* Every nest loop other than the specials is the output channel (in [d] and [b]), the reduction
       channel (in [a] and [b]) or a batch axis (in [d] and [a]); every symbol occurs at most once
       per map; and a symbol not bound by the nest (a static index) never enters [b]. *)
    let no_dups l = not (List.contains_dup l ~compare:Idx.compare_symbol) in
    no_dups d_o && no_dups a_o && no_dups b_o && List.for_all b_o ~f:is_loop
    && List.for_all loop_syms ~f:(fun s ->
        is_special s
        || (mem d_o s && mem b_o s && not (mem a_o s))
        || (mem a_o s && mem b_o s && not (mem d_o s))
        || (mem d_o s && mem a_o s && not (mem b_o s)))
  in
  let channel ~in1 ~in2 ~out =
    List.filter loop_syms ~f:(fun s -> mem in1 s && mem in2 s && not (mem out s))
  in
  let* ic = match channel ~in1:a_o ~in2:b_o ~out:d_o with [ ic ] -> Some ic | _ -> None in
  let one_oc = List.length (channel ~in1:d_o ~in2:b_o ~out:a_o) = 1 in
  let all_mentioned = List.for_all loop_syms ~f:(fun s -> is_special s || mem d_o s || mem a_o s) in
  let f32 (tn : Tn.t) = Ops.equal_prec (Lazy.force tn.Tn.storage_prec) Ops.single in
  if
    (not (Tn.equal a d))
    && (not (Tn.equal b d))
    && (not (Tn.equal a b))
    && loops_ok && separable && all_plain di ~except:[] && all_plain bi ~except:[]
    && all_plain ai ~except:[ h.a_axis; w.a_axis ]
    && roles_ok () && one_oc && all_mentioned && f32 d && f32 a && f32 b
  then
    let prec = Ops.single in
    let key =
      (* Loop symbol [i] of the nest becomes [Symbol (-1 - i)], which no lowering mints; a symbol
         the nest does not bind (a static index) stays itself, as it is stable across lowerings. *)
      let canon sym =
        match List.findi loop_syms ~f:(fun _ s -> Idx.equal_symbol s sym) with
        | Some (i, _) -> Idx.Symbol (-1 - i)
        | None -> sym
      in
      let canon_idx (idcs : Idx.axis_index array) =
        Array.map idcs ~f:(function
          | Idx.Iterator sym -> Idx.Iterator (canon sym)
          | Idx.Affine { symbols; offset } ->
              Idx.affine ~offset ~symbols:(List.map symbols ~f:(fun (c, sym) -> (c, canon sym)))
          | idx -> idx)
      in
      Sexp.to_string
        ([%sexp_of:
           int
           * Idx.axis_index array
           * int
           * Idx.axis_index array
           * int
           * Idx.axis_index array
           * int list]
           ( d.Tn.uid,
             canon_idx di,
             a.Tn.uid,
             canon_idx ai,
             b.Tn.uid,
             canon_idx bi,
             List.map loops ~f:(fun l -> l.to_) ))
    in
    Some { loops; d; di; a; ai; b; bi; prec; h; w; ic; rebuild; key }
  else None

(* {1 Emission} *)

let subst_idx (m : (Idx.symbol * Idx.axis_index) list) (idcs : Idx.axis_index array) =
  Array.map idcs ~f:(function
    | Idx.Iterator s as idx -> (
        match List.Assoc.find m s ~equal:Idx.equal_symbol with Some i -> i | None -> idx)
    | idx -> idx)

(* A linear combination of reads with constant coefficients, zero terms dropped, as a left-nested
   chain of additions and subtractions at the site's precision. *)
let lin_comb prec (terms : (float * LL.scalar_t) list) : LL.scalar_t =
  let terms = List.filter terms ~f:(fun (c, _) -> not (Float.equal c 0.)) in
  let scaled c x =
    if Float.equal c 1. then x else LL.Binop (Ops.Mul, (LL.Constant c, prec), (x, prec))
  in
  (* Lead with a positive term where there is one, so no negation is emitted. *)
  let terms =
    match List.partition_tf terms ~f:(fun (c, _) -> Float.(c > 0.)) with
    | p :: ps, ns -> (p :: ps) @ ns
    | [], ns -> ns
  in
  match terms with
  | [] -> LL.Constant 0.
  | (c, x) :: rest ->
      List.fold rest ~init:(scaled c x) ~f:(fun acc (c, x) ->
          if Float.(c < 0.) then LL.Binop (Ops.Sub, (acc, prec), (scaled (Float.neg c) x, prec))
          else LL.Binop (Ops.Add, (acc, prec), (scaled c x, prec)))

let seq_of = LL.unflat_lines

(* Loops in the given order around [body], each over [0, n - 1]. *)
let nest_around (loops : (Idx.symbol * int) list) body =
  List.fold_right loops ~init:body ~f:(fun (index, n) body ->
      LL.For_loop { index; from_ = 0; to_ = n - 1; body; axis = LL.Serial })

let rewrite_site (s : site) : LL.t =
  let p = s.prec in
  let ext sym =
    List.find_map_exn s.loops ~f:(fun l ->
        if Idx.equal_symbol l.index sym then Some (l.to_ + 1) else None)
  in
  let nth = s.h.n_o / 2 and ntw = s.w.n_o / 2 in
  let is_o sym = Idx.equal_symbol sym s.h.o || Idx.equal_symbol sym s.w.o in
  let is_k sym = Idx.equal_symbol sym s.h.k || Idx.equal_symbol sym s.w.k in
  (* The loop symbols of a map, in the map's axis order: the axes a minted node keeps. Fixed and
     static axes are constant over the nest and dropped. *)
  let loop_axes idcs =
    Array.to_list idcs
    |> List.filter_map ~f:(function
      | Idx.Iterator sym when List.exists s.loops ~f:(fun l -> Idx.equal_symbol l.index sym) ->
          Some sym
      | Idx.Affine _ as idx ->
          (* A conv axis of [a]: kept as its output iterator, tiled. *)
          if Idx.axis_index_mentions_symbol s.h.o idx then Some s.h.o
          else if Idx.axis_index_mentions_symbol s.w.o idx then Some s.w.o
          else None
      | _ -> None)
  in
  (* A tiled extent: an output iterator's axis holds tiles. *)
  let tiled_ext sym =
    if Idx.equal_symbol sym s.h.o then nth else if Idx.equal_symbol sym s.w.o then ntw else ext sym
  in
  let u_axes = List.filter (loop_axes s.bi) ~f:(fun x -> not (is_k x)) in
  let v_axes = loop_axes s.ai in
  let m_axes = loop_axes s.di in
  let dims axes = Array.of_list (4 :: 4 :: List.map axes ~f:tiled_ext) in
  let u = mint ~site_key:s.key ~role:`Kernel ~like:s.b p (dims u_axes) in
  let v = mint ~site_key:s.key ~role:`Input ~like:s.a p (dims v_axes) in
  let m = mint ~site_key:s.key ~role:`Product ~like:s.d p (dims m_axes) in
  (* Every nest binds fresh symbols: a symbol bound by two sibling loops makes the routine
     uncacheable. [fresh axes] maps each original symbol of [axes] to a new one. *)
  let fresh axes = List.map axes ~f:(fun sym -> (sym, Idx.get_symbol ())) in
  let it m sym = Idx.Iterator (List.Assoc.find_exn m sym ~equal:Idx.equal_symbol) in
  let node_idx m ~xi ~nu axes = Array.of_list (xi :: nu :: List.map axes ~f:(it m)) in
  let in_order axes =
    (* The original loop order, restricted to [axes]. *)
    List.filter_map s.loops ~f:(fun l ->
        if List.mem axes l.index ~equal:Idx.equal_symbol then Some l.index else None)
  in
  let loops_of fm axes =
    List.map axes ~f:(fun sym ->
        (List.Assoc.find_exn fm sym ~equal:Idx.equal_symbol, tiled_ext sym))
  in
  let fix i = Idx.Fixed_idx i in
  let set tn idcs llsc = LL.Set { tn; idcs; llsc; debug = "" } in
  let get tn idcs = LL.Get (tn, idcs) in
  (* 1. The kernel transform [U = G w G^T], per output and reduction channel. *)
  let kernel =
    let fm = fresh u_axes in
    let sub = List.map fm ~f:(fun (o, n) -> (o, Idx.Iterator n)) in
    let body =
      List.concat_map (List.range 0 4) ~f:(fun xi ->
          List.map (List.range 0 4) ~f:(fun nu ->
              let terms =
                List.concat_map (List.range 0 3) ~f:(fun i ->
                    List.map (List.range 0 3) ~f:(fun j ->
                        let idcs = subst_idx ((s.h.k, fix i) :: (s.w.k, fix j) :: sub) s.bi in
                        (g.(xi).(i) *. g.(nu).(j), get s.b idcs)))
              in
              set u (node_idx fm ~xi:(fix xi) ~nu:(fix nu) u_axes) (lin_comb p terms)))
    in
    nest_around (loops_of fm (in_order u_axes)) (seq_of body)
  in
  (* 2. The input transform [V = B^T x B], per 4x4 tile at [(2 th, 2 tw)]. *)
  let input =
    let fm = fresh v_axes in
    let sub =
      List.filter_map fm ~f:(fun (o, n) -> if is_o o then None else Some (o, Idx.Iterator n))
    in
    let tile cx i =
      Idx.affine ~symbols:[ (2, List.Assoc.find_exn fm cx.o ~equal:Idx.equal_symbol) ] ~offset:i
    in
    let body =
      List.concat_map (List.range 0 4) ~f:(fun xi ->
          List.map (List.range 0 4) ~f:(fun nu ->
              let terms =
                List.concat_map (List.range 0 4) ~f:(fun i ->
                    List.map (List.range 0 4) ~f:(fun j ->
                        let idcs = subst_idx sub s.ai in
                        idcs.(s.h.a_axis) <- tile s.h i;
                        idcs.(s.w.a_axis) <- tile s.w j;
                        (b_t.(xi).(i) *. b_t.(nu).(j), get s.a idcs)))
              in
              set v (node_idx fm ~xi:(fix xi) ~nu:(fix nu) v_axes) (lin_comb p terms)))
    in
    nest_around (loops_of fm (in_order v_axes)) (seq_of body)
  in
  (* 3. The 16 per-position contractions over the reduction channel: the write loops in [d]'s axis
     order (so the output channel, [d]'s minor axis, is the innermost write loop), the reduction
     innermost -- the lowered shape of a product, for the matmul sketches. *)
  let product =
    let xi = Idx.get_symbol () and nu = Idx.get_symbol () in
    let fm = fresh (m_axes @ [ s.ic ]) in
    let xi_i = Idx.Iterator xi and nu_i = Idx.Iterator nu in
    let mi = node_idx fm ~xi:xi_i ~nu:nu_i m_axes in
    let rhs =
      LL.Binop
        ( Ops.Add,
          (get m mi, p),
          ( LL.Binop
              ( Ops.Mul,
                (get v (node_idx fm ~xi:xi_i ~nu:nu_i v_axes), p),
                (get u (node_idx fm ~xi:xi_i ~nu:nu_i u_axes), p) ),
            p ) )
    in
    seq_of
      [
        LL.Zero_out m;
        nest_around ((xi, 4) :: (nu, 4) :: loops_of fm (m_axes @ [ s.ic ])) (set m mi rhs);
      ]
  in
  (* 4. The inverse transform [Y = A^T M A] into each 2x2 output tile, accumulated into [d] so its
     prior contents (the zeroing ahead of the nest, or anything else) keep their meaning. *)
  let output =
    let fm = fresh m_axes in
    let sub =
      List.filter_map fm ~f:(fun (o, n) -> if is_o o then None else Some (o, Idx.Iterator n))
    in
    let body =
      List.concat_map (List.range 0 2) ~f:(fun al ->
          List.map (List.range 0 2) ~f:(fun be ->
              let at cx off =
                Idx.affine
                  ~symbols:[ (2, List.Assoc.find_exn fm cx.o ~equal:Idx.equal_symbol) ]
                  ~offset:off
              in
              let di = subst_idx ((s.h.o, at s.h al) :: (s.w.o, at s.w be) :: sub) s.di in
              let terms =
                List.concat_map (List.range 0 4) ~f:(fun xi ->
                    List.map (List.range 0 4) ~f:(fun nu ->
                        ( a_t.(al).(xi) *. a_t.(be).(nu),
                          get m (node_idx fm ~xi:(fix xi) ~nu:(fix nu) m_axes) )))
              in
              set s.d di (LL.Binop (Ops.Add, (get s.d di, p), (lin_comb p terms, p)))))
    in
    nest_around (loops_of fm m_axes) (seq_of body)
  in
  (* 5. Odd output extents: the last row, then the last column of the rows above it, in the direct
     form -- the original statement over fresh symbols and restricted ranges, disjoint strips. *)
  let strip ~h_range ~w_range =
    let fm = fresh (List.map s.loops ~f:(fun l -> l.index)) in
    let sub = List.map fm ~f:(fun (o, n) -> (o, Idx.Iterator n)) in
    let f idcs =
      Array.map idcs ~f:(function
        | Idx.Iterator _ as idx -> (subst_idx sub [| idx |]).(0)
        | Idx.Affine { symbols; offset } ->
            Idx.affine ~offset
              ~symbols:
                (List.map symbols ~f:(fun (c, sym) ->
                     (c, Option.value ~default:sym (List.Assoc.find fm sym ~equal:Idx.equal_symbol))))
        | idx -> idx)
    in
    let body = set s.d (f s.di) (s.rebuild f) in
    List.fold_right s.loops ~init:body ~f:(fun l body ->
        let index = List.Assoc.find_exn fm l.index ~equal:Idx.equal_symbol in
        let from_, to_ =
          if Idx.equal_symbol l.index s.h.o then h_range
          else if Idx.equal_symbol l.index s.w.o then w_range
          else (l.from_, l.to_)
        in
        LL.For_loop { index; from_; to_; body; axis = LL.Serial })
  in
  let strips =
    (if s.h.n_o % 2 = 1 then [ strip ~h_range:(s.h.n_o - 1, s.h.n_o - 1) ~w_range:(0, s.w.n_o - 1) ]
     else [])
    @
    if s.w.n_o % 2 = 1 then
      [ strip ~h_range:(0, (2 * nth) - 1) ~w_range:(s.w.n_o - 1, s.w.n_o - 1) ]
    else []
  in
  seq_of ([ kernel; input; product; output ] @ strips)

let rewrite (llc : LL.t) : LL.t =
  let stmts = LL.flat_lines [ llc ] in
  let changed = ref false in
  let stmts =
    List.map stmts ~f:(fun stmt ->
        match find_site stmt with
        | Some site ->
            changed := true;
            rewrite_site site
        | None -> stmt)
  in
  if !changed then LL.unflat_lines stmts else llc
