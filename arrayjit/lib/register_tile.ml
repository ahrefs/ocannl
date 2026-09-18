open Base

type t = { rm : int; rn : int; lanes : int } [@@deriving sexp, compare, equal]

let to_string { rm; rn; lanes } = Printf.sprintf "rm%d rn%d lanes%d" rm rn lanes
let width { rn; lanes; _ } = rn * lanes
let live_registers { rm; rn; _ } = (rm * rn) + rm + rn

let simd_lane_ladder ~vector_bytes ~elt_bytes =
  if elt_bytes <= 0 || vector_bytes < 8 then []
  else
    let floor_bytes = min vector_bytes 32 in
    let rec ladder bytes =
      let lanes = bytes / elt_bytes in
      let rest = if bytes / 2 >= floor_bytes then ladder (bytes / 2) else [] in
      if lanes >= 2 then lanes :: rest else rest
    in
    ladder vector_bytes

let rm_cap = 4
let rn_cap ~vector_bytes = if vector_bytes = 32 then 3 else 6

let budget ~vector_bytes =
  let rn = rn_cap ~vector_bytes in
  (rm_cap * rn) + rm_cap + rn

(* The widths the register file renders, narrowed to those the column extent can fill. *)
let ladder_for ~vector_bytes ~elt_bytes ~n =
  simd_lane_ladder ~vector_bytes ~elt_bytes |> List.filter ~f:(fun lanes -> n >= lanes)

let default_rm ~m = min rm_cap m

(* The largest [rn] the budget admits beside [rm] rows: [rm * rn + rm + rn <= budget]. *)
let rn_budget_cap ~vector_bytes ~rm = (budget ~vector_bytes - rm) / (rm + 1)

type coverage = { m_full : int; n_full : int; tail_widths : int list }

let coverage ~m ~n t =
  let m_full = m - (m % t.rm) and n_full = n - (n % width t) in
  let n_tail = n - n_full in
  let tail_widths =
    List.init ((n_tail + t.lanes - 1) / t.lanes) ~f:(fun c -> min t.lanes (n_tail - (c * t.lanes)))
  in
  { m_full; n_full; tail_widths }

(* Per unit of m*k, in vector-issue slots: one fused FMA per vector column, plus the B row loads
   (1/rm of one per FMA) and the A splats (1/rn). The columns the width does not cover are a
   narrower tile of [tail_widths] vector columns (gh-ocannl-620) -- whole issues at less A-reuse,
   the last of them possibly partial -- never scalar code, so there is no fitted constant to
   carry. *)
let cost ~n ~rm ~lanes ~rn =
  let per_column rn = 1. +. (1. /. Float.of_int rm) +. (1. /. Float.of_int rn) in
  let { n_full; tail_widths; _ } = coverage ~m:rm ~n { rm; rn; lanes } in
  let rn_tail = List.length tail_widths in
  (Float.of_int (n_full / lanes) *. per_column rn)
  +. if rn_tail = 0 then 0. else Float.of_int rn_tail *. per_column rn_tail

let default ~vector_bytes ~elt_bytes ~m ~n =
  if m < 1 || n < 1 then None
  else
    let rm = default_rm ~m in
    let candidates =
      List.concat_map (ladder_for ~vector_bytes ~elt_bytes ~n) ~f:(fun lanes ->
          let cap = min (rn_cap ~vector_bytes) (n / lanes) in
          List.range 1 (cap + 1) |> List.map ~f:(fun rn -> { rm; rn; lanes }))
    in
    let tail t = List.length (coverage ~m ~n t).tail_widths in
    List.min_elt candidates ~compare:(fun t1 t2 ->
        (* Ties go to the wider vector (more work per issue), then to the tail-free tile (one tile
           body rather than two), then to the larger tile (more A-reuse). *)
        match
          Float.compare
            (cost ~n ~rm ~lanes:t1.lanes ~rn:t1.rn)
            (cost ~n ~rm ~lanes:t2.lanes ~rn:t2.rn)
        with
        | 0 -> (
            match Int.compare t2.lanes t1.lanes with
            | 0 -> (
                match Int.compare (tail t1) (tail t2) with 0 -> Int.compare t2.rn t1.rn | c -> c)
            | c -> c)
        | c -> c)

let check ~vector_bytes ~elt_bytes ~m ~n t =
  let ladder = ladder_for ~vector_bytes ~elt_bytes ~n in
  let full_ladder = simd_lane_ladder ~vector_bytes ~elt_bytes in
  if t.rm < 1 || t.rn < 1 || t.lanes < 1 then
    Error (Printf.sprintf "degenerate geometry %s" (to_string t))
  else if not (List.mem full_ladder t.lanes ~equal:Int.equal) then
    Error
      (Printf.sprintf
         "lanes=%d is not a width a %d-byte vector file renders at %d-byte elements (%s)" t.lanes
         vector_bytes elt_bytes
         (String.concat ~sep:"/" (List.map full_ladder ~f:Int.to_string)))
  else if not (List.mem ladder t.lanes ~equal:Int.equal) then
    Error (Printf.sprintf "n = %d below the vector width (lanes = %d)" n t.lanes)
  else if t.rm > m then Error (Printf.sprintf "rm=%d exceeds the row extent m=%d" t.rm m)
  else if width t > n then
    Error (Printf.sprintf "width rn*lanes=%d exceeds the column extent n=%d" (width t) n)
  else if live_registers t > budget ~vector_bytes then
    Error
      (Printf.sprintf
         "%d live registers (rm*rn + rm + rn) exceed the %d-register budget of a %d-byte vector \
          file"
         (live_registers t) (budget ~vector_bytes) vector_bytes)
  else Ok ()

let alternatives ~vector_bytes ~elt_bytes ~m ~n =
  match (default ~vector_bytes ~elt_bytes ~m ~n, ladder_for ~vector_bytes ~elt_bytes ~n) with
  | None, _ | _, [] -> []
  | Some dflt, lanes :: _ ->
      let rm = default_rm ~m in
      let cap = min (rn_budget_cap ~vector_bytes ~rm) (n / lanes) in
      (* Among tail-free widths the model's ranking is exact -- equal issue counts, strictly more
         A-reuse at the larger [rn] -- so only the largest is worth a timing; the smaller ones are
         dominated on its own terms. What it cannot see is how a tail-bearing default's second tile
         really costs, which is what that one twin measures. *)
      let tail_free =
        List.range 2 (cap + 1) |> List.filter ~f:(fun rn -> n % (rn * lanes) = 0) |> List.last
      in
      (* The budget cap joins only when its column tail is at most one vector: a fatter tail is a
         narrower second tile whose lower A-reuse the model already prices, and seeding the cap on
         every leaf doubled the CPU tensorized seed count for candidates the tuner rejects. gh-614's
         n = 512 AVX2 site (rn = 3, a one-vector tail) stays in. *)
      let budget_cap =
        if
          cap >= 2
          && (not (Option.equal Int.equal tail_free (Some cap)))
          && n % (cap * lanes) <= lanes
        then Some cap
        else None
      in
      let rns = List.filter_opt [ tail_free; budget_cap ] in
      List.map rns ~f:(fun rn -> { rm; rn; lanes })
      |> List.filter ~f:(fun t -> not (equal t dflt))
      |> List.filter ~f:(fun t -> Result.is_ok (check ~vector_bytes ~elt_bytes ~m ~n t))
