open Base
module Tn = Ir.Tnode
module Tensor = Ocannl_tensor.Tensor

type progress = {
  step : int;
  epoch : int;
  position : int;
  host_rng : Stdlib.Random.State.t option;
  extra : (string * string) list;
}

let carried_state (routine : Context.routine) =
  if List.is_empty routine.Context.segments then
    invalid_arg
      ("Checkpoint.carried_state: routine " ^ routine.Context.name
     ^ " records no kernel segments, so what it carries between runs cannot be read");
  (* The segments' traced stores are slices of the routine's one store: their flags are routine-wide
     verdicts ([read_before_write] is about the routine's entry values, not the segment's), so the
     union over segments is the routine's own interface. A node is carried when the routine both
     consumes its entry value and writes it: a parameter, an optimizer moment, an accumulator. A
     node it only reads (a dataset, a host-set learning rate) is an input the program
     re-establishes, and one it writes before reading (a gradient zeroed each step) carries
     nothing. *)
  let inputs, outputs =
    List.fold routine.Context.segments
      ~init:(Set.empty (module Tn), Set.empty (module Tn))
      ~f:(fun (ins, outs) seg ->
        let (seg_ins, seg_outs), _merge = Ir.Low_level.input_and_output_nodes seg in
        (Set.union ins seg_ins, Set.union outs seg_outs))
  in
  Set.inter inputs outputs

let training_state ?(extra = []) ~routines loss =
  let params =
    Set.to_list loss.Tensor.params
    |> List.map ~f:(fun p -> p.Tensor.value)
    |> Set.of_list (module Tn)
  in
  List.fold routines
    ~init:(Set.union params (Set.of_list (module Tn) extra))
    ~f:(fun acc r -> Set.union acc (carried_state r))

(* The header keys this module owns. The user's [extra] pairs live under their own prefix, so no
   choice of user key can shadow one of these. *)
let format_key = "ocannl.training_checkpoint"
let format_version = "1"
let step_key = "ocannl.step"
let epoch_key = "ocannl.epoch"
let position_key = "ocannl.position"
let host_rng_key = "ocannl.host_rng"
let state_key = "ocannl.state"
let extra_prefix = "extra."

(* The state set's identity, as the checkpoint matches nodes: by (namespace, id). Recorded so that a
   resumed program deriving a DIFFERENT state set is refused rather than partially restored -- a set
   missing the optimizer moments would restore the parameters and silently restart the moments from
   their initialization, which is a different run. *)
let state_keys state =
  Set.to_list state
  |> List.map ~f:(fun tn -> tn.Tn.namespace ^ ":" ^ Int.to_string tn.Tn.id)
  |> List.sort ~compare:String.compare

let hex_of_string s = String.concat_map s ~f:(fun c -> Printf.sprintf "%02x" (Char.to_int c))

let string_of_hex ~path h =
  let n = String.length h in
  if n % 2 <> 0 then failwith ("Checkpoint.restore: " ^ path ^ ": malformed host RNG state");
  String.init (n / 2) ~f:(fun i ->
      match Int.of_string_opt ("0x" ^ String.sub h ~pos:(2 * i) ~len:2) with
      | Some b -> Char.of_int_exn b
      | None -> failwith ("Checkpoint.restore: " ^ path ^ ": malformed host RNG state"))

let save ~ctx ~progress state path =
  if Set.is_empty state then invalid_arg "Checkpoint.save: the training state is empty";
  if List.contains_dup progress.extra ~compare:(fun (k1, _) (k2, _) -> String.compare k1 k2) then
    invalid_arg "Checkpoint.save: duplicate keys in progress.extra";
  let metadata =
    [
      (format_key, format_version);
      (step_key, Int.to_string progress.step);
      (epoch_key, Int.to_string progress.epoch);
      (position_key, Int.to_string progress.position);
      (state_key, String.concat ~sep:" " (state_keys state));
    ]
    @ Option.value_map progress.host_rng ~default:[] ~f:(fun rng ->
        [ (host_rng_key, hex_of_string (Stdlib.Random.State.to_binary_string rng)) ])
    @ List.map progress.extra ~f:(fun (k, v) -> (extra_prefix ^ k, v))
  in
  Persistence.save ~ctx ~appending:false ~metadata state path

let read_progress path metadata =
  let find key = List.Assoc.find metadata key ~equal:String.equal in
  (match find format_key with
  | Some v when String.equal v format_version -> ()
  | Some v ->
      failwith
        ("Checkpoint.restore: " ^ path ^ ": unsupported training-checkpoint version " ^ v
       ^ " (this build reads version " ^ format_version ^ ")")
  | None ->
      failwith
        ("Checkpoint.restore: " ^ path
       ^ " is not a training checkpoint: it records no progress (a plain Persistence.save writes \
          tensors only; use Persistence.restore for it)"));
  let int_field key =
    match Option.bind (find key) ~f:Int.of_string_opt with
    | Some v -> v
    | None -> failwith ("Checkpoint.restore: " ^ path ^ ": missing or malformed " ^ key)
  in
  let progress =
    {
      step = int_field step_key;
      epoch = int_field epoch_key;
      position = int_field position_key;
      host_rng =
        Option.map (find host_rng_key) ~f:(fun h ->
            Stdlib.Random.State.of_binary_string (string_of_hex ~path h));
      extra =
        List.filter_map metadata ~f:(fun (k, v) ->
            Option.map (String.chop_prefix k ~prefix:extra_prefix) ~f:(fun k -> (k, v)));
    }
  in
  let recorded = Option.value_map (find state_key) ~default:[] ~f:(String.split ~on:' ') in
  (progress, List.filter recorded ~f:(Fn.non String.is_empty))

let restore ~ctx state path =
  let given = state_keys state in
  let progress = ref None in
  (* Validated inside the restore, from the header it reads the payloads with: one snapshot of the
     file, and nothing uploaded when a check fails. *)
  let check metadata =
    let p, recorded = read_progress path metadata in
    if not (List.equal String.equal recorded given) then begin
      let recorded_set = Set.of_list (module String) recorded in
      let given_set = Set.of_list (module String) given in
      let show s = if Set.is_empty s then "none" else String.concat ~sep:" " (Set.to_list s) in
      failwith
        ("Checkpoint.restore: " ^ path
       ^ ": the training state differs from the one saved, so the run would not continue as it was \
          (derive it from the same model and routines). Saved only: "
        ^ show (Set.diff recorded_set given_set)
        ^ "; given only: "
        ^ show (Set.diff given_set recorded_set))
    end;
    progress := Some p
  in
  let ctx = Persistence.restore ~ctx ~check_metadata:check state path in
  match !progress with
  | Some p -> (ctx, p)
  | None -> failwith "Checkpoint.restore: internal error: the checkpoint header was not checked"
