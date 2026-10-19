(* The two-matmul chain the scripted-ranking autotune tests search ([autotune_composite_playoff],
   [autotune_round_expansion]), and the small seeding menu that keeps their searches cheap
   (lukstafi/ocannl-staging#1143) -- and the searches of [autotune_sketch_placements], whose own
   chain also fissions into two keyed matmul segments on cc.

   The chain fissions into two keyed matmul segments of different shapes. Unpruned, each segment
   offers 21 per-segment sketch singles, and a search compiles every single it times: those compiles
   are such a test's whole cost (about 25 s of CPU per search on mac-studio). [small_menu] keeps a
   fifth of each segment's sketches by the roofline model -- [autotune_keep_fraction=0.2] with a
   synthetic envelope ([model_peak_flops], [model_peak_memory_bandwidth]), since cc advertises none
   and without an envelope the pre-filter keeps everything -- which leaves eight and seven singles
   per segment. The envelope only ranks a segment's sketches against each other, so it need not
   describe the machine.

   Which singles survive is the model's choice, and no claim may depend on it. How many survive is
   what the claims depend on: a width-2 playoff claim, or a beam whose second entry is a playoff
   alternate, cannot fail over a segment offering one or two. So a test that prunes states
   [menu_claim] over what its search timed, and a cost-model change that ever kept fewer fails that
   claim loudly instead of the others passing vacuously.

   Which singles survive also follows the cc vector width, which auto-probes from the host: at 32
   bytes (AVX2) the pre-filter keeps eight and seven singles, at 64 (AVX-512) four and three -- the
   latter on the [menu_claim] floor -- and CI's x86 runners draw either CPU at random. So
   [small_menu] pins the width ([vector_bytes]) for every search of the process, and with it the
   register count ([vector_registers]), which the register tile's column cap follows
   (gh-ocannl-1285) and which an AVX-512 runner keeps at 32 under the pinned 32-byte width; and
   [menu_claim] prints the width and count the backend resolved beside the menu it reports, so a log
   names which menu it saw.

   The envelope overrides are read once per process, at the first pre-filter, so [small_menu] (or
   [small_menu_envelope]) runs at a test's top level before its first search, at the config-file
   level; the test's stanza declares the keys' environment variables, so an ambient override reruns
   it rather than serving the previous result. *)
open Base
open Ocannl
open Ocannl.Operation.DSL_modules
open Verdict.Claims

let q = 32

let matrix ~label ~modulus ~offset ~stride =
  TDSL.ndarray
    (Array.init (q * q) ~f:(Ll_test.cycle_flat ~dims:[| q; q |] ~modulus ~offset ~stride))
    ~label:[ label ] ~input_dims:[ q ] ~output_dims:[ q ] ()

(* Two matmul segments of different shapes (the multi-site chain of [autotune_fission_sketch]): the
   materialized sums cut the chain so each product is a keyed segment of its own. *)
let chain_with_output () =
  let qa = matrix ~label:"qa" ~modulus:11 ~offset:0. ~stride:0.125 in
  let qb = matrix ~label:"qb" ~modulus:7 ~offset:(-3.) ~stride:1. in
  let qc = matrix ~label:"qc" ~modulus:5 ~offset:(-2.) ~stride:0.5 in
  let qc16 =
    TDSL.ndarray
      (Array.init (16 * q)
         ~f:(Ll_test.cycle_flat ~dims:[| 16; q |] ~modulus:9 ~offset:(-4.) ~stride:0.25))
      ~label:[ "qc16" ] ~input_dims:[ q ] ~output_dims:[ 16 ] ()
  in
  let%op qd = qa + qb in
  Train.set_materialized qd.Tensor.value;
  let%op qe = qd * qc in
  Train.set_materialized qe.Tensor.value;
  let%op qg = qc16 * qe in
  (qg, Train.forward qg)

(* The chain's computation alone. [chain_with_output] also returns the output tensor, for a test
   that reads what a crowned routine computes. *)
let chain () = snd (chain_with_output ())

(* The fraction of each segment's sketches the small menu keeps, for a test that passes it to one
   search ([Autotune.tune ~keep_fraction]) rather than stating it for all of them. *)
let keep_fraction = 0.2

let state settings =
  List.iter settings ~f:(fun (key, data) -> Hashtbl.set Utils.config_file_args ~key ~data)

(* The envelope alone: what a test that prunes only some of its searches states. *)
let small_menu_envelope () =
  state [ ("model_peak_flops", "1e12"); ("model_peak_memory_bandwidth", "1e11") ]

(* The cc vector file the small menu is pinned to (see the header): AVX2's, 32 bytes on 16
   registers, where each segment keeps more than the claims' floor. *)
let vector_bytes = 32
let vector_registers = 16

(* The envelope, the keep fraction, the cc vector file and the pad-waste economy, for every search
   of the process. The chain's 32-wide products sit below the CPU packed menu's 64- to 256-deep
   blocks, so the pad-waste economy (gh-ocannl-740) would exclude most of each segment's sketches
   and leave a menu other than the one the claims built on it were tuned against -- the round
   expansion's positional control among them. These tests are about the playoff, the round and the
   placement arms over a fixed menu, not about which geometries a pad makes worth proposing, so the
   small menu proposes every pad ([inf]) and keeps its fifth from the menu it always had. *)
let small_menu () =
  small_menu_envelope ();
  state
    [
      ("autotune_keep_fraction", Float.to_string keep_fraction);
      ("cc_vector_bytes", Int.to_string vector_bytes);
      ("cc_vector_registers", Int.to_string vector_registers);
      ("autotune_pad_waste_max", "inf");
    ]

(* A per-segment sketch label's focus (gh-ocannl-1138): what precedes [" | staff "], where a staffed
   single lists the running composite's choices for the other keys. A bare label is all focus. *)
let focus label =
  match String.substr_index label ~pattern:" | staff " with
  | Some i -> String.prefix label i
  | None -> label

(* Single or composite by the focus alone: a staffed single schedules several segments but is a
   single of its focus key. *)
let is_single label =
  String.is_prefix label ~prefix:"F_sketch[" && not (String.mem (focus label) ',')

let is_composite label = String.is_prefix label ~prefix:"F_sketch[" && String.mem (focus label) ','

let is_staffed label =
  String.is_prefix label ~prefix:"F_sketch[" && String.is_substring label ~substring:" | staff "

(* A single's keyed segment: the tag its focus ends with ([@<digests>:u<n>]). *)
let segment_of label =
  List.last_exn (String.split (String.chop_suffix_if_exists (focus label) ~suffix:"]") ~on:'@')

(* The singles among [measured] (a search's timed labels, in attempt order), counted per keyed
   segment. *)
let singles_per_segment measured =
  List.filter measured ~f:is_single
  |> List.sort_and_group ~compare:(fun a b -> String.compare (segment_of a) (segment_of b))
  |> List.map ~f:List.length

(* The claims a pruned search's menu owes the claims built on it: the search keyed two segments
   ([chain]'s two products; [autotune_sketch_placements]' structural twins and closing layer), and
   each still offers three or more singles. A test that prunes on some backends only passes [~when_]
   (whether this run pruned) and [~on] (the backend), and the claims are skipped where it did not.
   Where it pruned, the report names the cc vector width the backend resolved, which the menu
   follows. *)
let menu_claim ?(when_ = true) ?(on = "") measured =
  let per_segment = singles_per_segment measured in
  Stdio.eprintf "menu (not part of the golden): singles per segment %s%s\n%!"
    (String.concat ~sep:" " (List.map per_segment ~f:Int.to_string))
    (if when_ then
       Printf.sprintf " at cc_vector_bytes=%d cc_vector_registers=%d"
         (Context.Cc_backend.vector_bytes_setting ())
         (Context.Cc_backend.vector_registers_setting ())
     else "");
  gated ~when_ ~on "menu: the pre-filtered seeding keys two segments" (List.length per_segment = 2);
  gated_all ~when_ ~on "menu: each segment offers three or more singles" per_segment ~f:(fun n ->
      n >= 3)
