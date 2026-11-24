(* gh-ocannl-1908: every benchmark model builds under the default [op_shape_schemes=check], with the
   same shapes as under [off]. [@train] passed while cifar_resnet died under [check]
   (gh-ocannl-1820), because no test built a downsampling residual model with the schemes on.

   Each benchmark workload is built from a synthetic fixture with its metadata and zero-valued
   tensors of the layouts the model reads. The runner's graph is built once per mode in a fresh
   session, and every tensor's dims are compared. Nn_blocks.resnet_block at stride 2 joins them as
   the library's own downsampling block. *)

open Base
open Ocannl
open Nn_blocks.DSL_modules
open Verdict.Claims
module St = Safetensors
module IDX = Train.IDX
module Row = Ocannl.Row

let workloads_dir = "../../benchmarks/workloads"

let specs =
  Stdlib.Sys.readdir workloads_dir |> Array.to_list
  |> List.filter ~f:(String.is_suffix ~suffix:".json")
  |> List.sort ~compare:String.compare
  |> List.map ~f:(fun f -> Yojson.Safe.from_file (Stdlib.Filename.concat workloads_dir f))

let field spec k =
  match spec with `Assoc kv -> List.Assoc.find kv ~equal:String.equal k | _ -> None

let int_field ?default spec k =
  match (field spec k, default) with
  | Some (`Int n), _ -> n
  | None, Some d -> d
  | _ -> failwith ("bench_model_schemes: no integer field " ^ k)

let str_field ~default spec k = match field spec k with Some (`String s) -> s | _ -> default

(* gen_fixtures.py's metadata: each spec value as a string, [stage_channels] comma-joined; the keys
   the builders derive ([mode], [use_padding], [stride1], ...) default the same way when absent. *)
let metadata spec =
  match spec with
  | `Assoc kv ->
      List.filter_map kv ~f:(fun (k, v) ->
          match v with
          | `Int n -> Some (k, Int.to_string n)
          | `Float x -> Some (k, Float.to_string x)
          | `Bool b -> Some (k, Bool.to_string b)
          | `String s -> Some (k, s)
          | `List l ->
              Some
                ( k,
                  String.concat ~sep:","
                    (List.map l ~f:(function
                      | `Int n -> Int.to_string n
                      | _ -> failwith ("bench_model_schemes: non-integer list " ^ k))) )
          | _ -> None)
  | _ -> failwith "bench_model_schemes: a workload spec is not an object"

(* The batch count only sets the batch symbol's range, so two keep the fixtures small. *)
let n_batches spec = Int.min 2 (int_field spec "n_batches")

(* The tensors each model reads, in gen_fixtures.py's layouts. *)
let fixture_tensors spec =
  let total = int_field spec "batch_size" * n_batches spec in
  match str_field ~default:"mlp" spec "model" with
  | "mlp" ->
      let dims =
        match field spec "dims" with
        | Some (`List dims) -> List.map dims ~f:Yojson.Safe.Util.to_int
        | _ -> failwith "bench_model_schemes: missing MLP dims"
      in
      let layers = List.zip_exn (List.drop_last_exn dims) (List.tl_exn dims) in
      [ ("x", [ total; List.hd_exn dims ]); ("y", [ total; List.last_exn dims ]) ]
      @ List.concat_mapi layers ~f:(fun i (din, dout) ->
          let n = i + 1 in
          [ (Printf.sprintf "w%d" n, [ dout; din ]); (Printf.sprintf "b%d" n, [ dout ]) ])
  | "conv" ->
      let img = int_field spec "image_size" in
      [
        ("x", [ total; img; img; int_field ~default:1 spec "in_channels" ]);
        ("y", [ total; int_field spec "classes" ]);
      ]
  | "gpt" ->
      let d = int_field spec "d_model" and v = int_field spec "vocab" in
      let seq = int_field spec "seq_len" and dff = int_field spec "d_ff" in
      [ ("wte", [ d; v ]); ("wpe", [ seq; d ]); ("ids", [ total; seq ]); ("tgt", [ total; seq ]) ]
      @ List.concat_map
          (List.range 0 (int_field spec "n_layer"))
          ~f:(fun i ->
            let n w = Printf.sprintf "l%d_%s" i w in
            [
              (n "ffn_w1", [ dff; d ]);
              (n "ffn_b1", [ dff ]);
              (n "ffn_w2", [ d; dff ]);
              (n "ffn_b2", [ d ]);
            ])
  | model -> failwith ("bench_model_schemes: no fixture for model " ^ model)

(* A safetensors file of F32 zeros: the 8-byte header length, the JSON header padded to an 8-byte
   boundary (so the payloads map rather than decode), the payloads back to back. *)
let write_fixture ~meta tensors =
  let path = Stdlib.Filename.temp_file "bench_model_schemes" ".safetensors" in
  let size = ref 0 in
  let entries =
    List.map tensors ~f:(fun (name, shape) ->
        let start = !size in
        size := start + (4 * List.fold shape ~init:1 ~f:( * ));
        ( name,
          `Assoc
            [
              ("dtype", `String "F32");
              ("shape", `List (List.map shape ~f:(fun d -> `Int d)));
              ("data_offsets", `List [ `Int start; `Int !size ]);
            ] ))
  in
  let meta = `Assoc (List.map meta ~f:(fun (k, v) -> (k, `String v))) in
  let header = Yojson.Safe.to_string (`Assoc (("__metadata__", meta) :: entries)) in
  let header = header ^ String.make ((8 - ((8 + String.length header) % 8)) % 8) ' ' in
  let len = Bytes.create 8 in
  Stdlib.Bytes.set_int64_le len 0 (Int64.of_int (String.length header));
  Stdio.Out_channel.with_file path ~binary:true ~f:(fun oc ->
      Stdio.Out_channel.output_bytes oc len;
      Stdio.Out_channel.output_string oc header;
      Stdio.Out_channel.output_string oc (String.make !size '\000'));
  path

(* bench_conv.ml's graph, up to the context: the batch slices, the model, the loss and the step's
   update code. *)
let build_conv st () =
  let batch_size = Bench_harness.meta_int st "batch_size" in
  let xs = TDSL.rebatch ~l:"xs" (St.to_ndarray st "x") () in
  let ys = TDSL.rebatch ~l:"ys" (St.to_ndarray st "y") () in
  let total = (Ir.Ndarray.dims (St.to_ndarray st "x")).(0) in
  let batch_n, _bindings = IDX.get_static_symbol ~static_range:(total / batch_size) IDX.empty in
  let%op batch_x = xs @| batch_n in
  let%op batch_y = ys @| batch_n in
  let { Bench_conv_model.model; mapping = _ } = Bench_conv_model.build st in
  let logits = model batch_x in
  let%op batch_loss =
    Nn_blocks.cross_entropy_loss ~spec:"...|v" ~normalize_by:!..batch_size () ~logits
      ~targets:batch_y
  in
  let learning_rate = TDSL.O.( !. ) (Float.of_string (Bench_harness.get_meta st "lr")) in
  ignore (Train.grad_update batch_loss : Ir.Assignments.comp);
  ignore (Train.sgd_update ~learning_rate batch_loss : Ir.Assignments.comp);
  batch_loss

let build_gpt st () =
  let training = Bench_harness.is_training st in
  let leg = Bench_harness.precision_leg ~runner:"bench_gpt" ~training ~st ~base:"f32" () in
  let { Bench_gpt_model.batch_loss; _ } = Bench_gpt_model.build_graph ~leg ~training st in
  ignore (Bench_gpt_model.build_step ~leg ~training st batch_loss);
  batch_loss

let build_mlp st () =
  (Bench_mlp_model.build_graph ~mp_prec:None ~no_slice:false ~materialize:false
     ~twin_placement:Mixed_prec.Twin_auto st)
    .batch_loss

(* Every tensor of the graph, in a deterministic walk from the root: its label and its dims. The
   first [Shape.to_dims] closes the session's inference. *)
let graph_dims root =
  let seen = Hash_set.create (module Int) in
  let acc = ref [] in
  let rec walk (t : Tensor.t) =
    if not (Hash_set.mem seen t.value.id) then (
      Hash_set.add seen t.value.id;
      acc := (Ir.Tnode.label_name t.value, Array.to_list (Shape.to_dims t.shape)) :: !acc;
      List.iter t.children ~f:(fun c -> walk c.subtensor))
  in
  walk root;
  Array.of_list (List.rev !acc)

let entry_to_string (name, dims) =
  name ^ " " ^ String.concat ~sep:"x" (List.map dims ~f:Int.to_string)

let entry_equal (n1, d1) (n2, d2) = String.equal n1 n2 && List.equal Int.equal d1 d2

(* One build in a fresh session under [mode]: the graph's dims, or the shape error, and how many
   scheme applications were checked. *)
let run ~mode build =
  Tensor.For_testing.set_op_shape_schemes (Some mode);
  Tensor.unsafe_reinitialize ();
  Exn.protect
    ~finally:(fun () -> Tensor.For_testing.set_op_shape_schemes None)
    ~f:(fun () ->
      let checked = ref 0 in
      let result =
        Tensor.For_testing.observe_schemes
          ~on_event:(function Tensor.Scheme.Checked _ -> Int.incr checked | _ -> ())
          (fun () ->
            match graph_dims (build ()) with
            | dims -> Ok dims
            | exception Row.Shape_error (msg, _) -> Error msg
            | exception Tensor.Session_error (msg, _) -> Error msg)
      in
      (result, !checked))

let parity_case name build =
  Verdict.case (name ^ ": builds under check, with off's shapes") (fun () ->
      let check, n_checked = run ~mode:`Check build in
      let off, _ = run ~mode:`Off build in
      Result.iter_error check
        ~f:(Stdio.eprintf "(not part of the golden) %s under check: %s\n%!" name);
      Result.iter_error off ~f:(Stdio.eprintf "(not part of the golden) %s under off: %s\n%!" name);
      p "under check, the model builds without a shape error" (Result.is_ok check);
      p "control (off): the model builds" (Result.is_ok off);
      p "under check, scheme applications were checked" (n_checked > 0);
      match (check, off) with
      | Ok dims_check, Ok dims_off ->
          Stdio.printf "%s: %d tensors\n" name (Array.length dims_off);
          p_all2 "check and off give every tensor of the graph the same label and dims" dims_check
            dims_off ~f:(fun a b ->
              if not (entry_equal a b) then
                Stdio.eprintf "(not part of the golden) %s: check %s, off %s\n%!" name
                  (entry_to_string a) (entry_to_string b);
              entry_equal a b)
      | _ -> ())

let () =
  let builders = [ ("conv", build_conv); ("gpt", build_gpt); ("mlp", build_mlp) ] in
  let built =
    List.map specs ~f:(fun spec ->
        let model = str_field ~default:"mlp" spec "model" in
        let build = List.Assoc.find_exn builders ~equal:String.equal model in
        let name = str_field ~default:"?" spec "name" in
        let meta = metadata spec in
        let meta =
          match (model, field spec "dims") with
          | "mlp", Some (`List dims) -> ("n_layers", Int.to_string (List.length dims - 1)) :: meta
          | _ -> meta
        in
        let path = write_fixture ~meta (fixture_tensors spec) in
        let st = St.read path in
        Exn.protect
          ~finally:(fun () ->
            St.close st;
            Stdlib.Sys.remove path)
          ~f:(fun () -> parity_case name (build st));
        model)
  in
  Verdict.case "the workload directory has each benchmark model" (fun () ->
      List.iter builders ~f:(fun (model, _) ->
          p_exists ("a " ^ model ^ " workload was built") built ~f:(String.equal model)))

(* [Nn_blocks.resnet_block] at stride 2: a strided 3x3 main path and a strided 1x1 downsample
   shortcut, with and without a channel change. *)
let () =
  List.iter [ Some 8; None ] ~f:(fun out_channels ->
      let name =
        "resnet_block stride 2"
        ^ Option.value_map out_channels ~default:"" ~f:(Printf.sprintf ", %d out channels")
      in
      parity_case name (fun () ->
          let block = Nn_blocks.resnet_block ~label:[ "rb" ] ~stride:2 ?out_channels () in
          let x =
            TDSL.range_of_shape ~label:[ "input" ] ~batch_dims:[ 2 ] ~output_dims:[ 8; 12; 3 ] ()
          in
          let%op loss = block ~train_step:None x ++ "...|... => 0" in
          loss))
