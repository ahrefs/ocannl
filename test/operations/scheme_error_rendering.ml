(* gh-ocannl-1799: a shape error raised inside a shape-scheme trace or check prints readably -- what
   [Printexc.to_string], and so the toplevel, shows for it. Each case is a tutorial-style mistake
   (test/operations/reusable_tensor_programs.ml, scheme_error_locality.ml): the rendering names the
   function, the conflicting sizes and the tensors involved with their axis kinds, and spells none
   of the reserved-range ids a trace or check draws its variables and tensors from.

   gh-ocannl-1903: an error's origins name the failing scheme's tensors first, and the cap
   (max_shape_error_origins) keeps those. The executable runs three times (test/operations/dune):
   plainly; with the cap at 3, where the application error's origins are trimmed; and with
   shape_error_full_sexp=true, the negative control of the claim that no rendering spells a
   reserved-range id. *)

open Base
open Ocannl.Nn_blocks.DSL_modules
open Verdict.Claims
module Row = Ocannl.Row

let input ?(batch = []) name output =
  Tensor.term ~grad_spec:Prohibit_grad ~label:[ name ] ~batch_dims:batch ~output_dims:output ()

let%op mlp_layer ~hid () x = relu (({ w; o = [ hid ] } * x) + { b = 0.; o = [ hid ] })

let%op residual_block ~hid () =
  let layer = mlp_layer ~hid () in
  fun x -> x + layer x

(* The same residual, opted out: nothing vouches for the layer, which names itself. *)
let%op residual_block_unchecked ~hid () =
  let layer = mlp_layer ~hid () in
  fun x -> x + layer x
[@@ocannl.no_scheme]

(* The second bias has the hidden size instead of the output size: no argument can fix it. *)
let%op two_layer ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }

let%op two_layer_unchecked ~hid ~out () x =
  let h = relu (({ w1; o = [ hid ] } * x) + { b1 = 0.; o = [ hid ] }) in
  ({ w2; o = [ out ] } * h) + { b2 = 0.; o = [ hid ] }
[@@ocannl.no_scheme]

let%op two_axes () x = x ++ "...|ab => ...|a"
let%op name_clash () x = x ++ "..v.., v => v, ..v.."
let%op concat_empty () x = x ++ "0^o => o"

(* Only the components' sum determines them: found when the closure forces the result's dims. *)
let%op concat_undetermined () x = x ++ "p^o => o"

(* The scheme that fails, where one does. *)
type case = { name : string; scheme : string option; run : unit -> unit }

let cases =
  [
    {
      name = "a mistake no argument can fix, at the trace";
      scheme = Some "two_layer";
      run = (fun () -> ignore (two_layer ~hid:8 ~out:2 () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "the same mistake, found by a definition check";
      scheme = Some "two_layer";
      run =
        (fun () ->
          ignore
            (Tensor.Scheme.check_definition ~name:"two_layer" (fun () ->
                 ignore (two_layer ~hid:8 ~out:2 () : Tensor.t -> Tensor.t))
              : Tensor.Scheme.t list));
    };
    {
      name = "an argument the scheme refuses, at the application";
      scheme = Some "residual_block";
      run = (fun () -> ignore (residual_block ~hid:4 () (input ~batch:[ 2 ] "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "a nested layer refusing a closed feature size: the outer function is named";
      scheme = Some "residual_block";
      run =
        (fun () ->
          let block = residual_block ~hid:4 () in
          ignore (Shape.to_dims (block (input ~batch:[ 2 ] "x" [ 4 ])).Tensor.shape : int array);
          ignore (block (input ~batch:[ 2 ] "y" [ 3 ]) : Tensor.t));
    };
    {
      name = "the same layer under an opted-out function: the layer is named";
      scheme = Some "mlp_layer";
      run =
        (fun () ->
          let block = residual_block_unchecked ~hid:4 () in
          ignore (Shape.to_dims (block (input ~batch:[ 2 ] "x" [ 4 ])).Tensor.shape : int array);
          ignore (block (input ~batch:[ 2 ] "y" [ 3 ]) : Tensor.t));
    };
    {
      name = "an argument with too few axes for the spec";
      scheme = Some "two_axes";
      run = (fun () -> ignore (two_axes () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "a dimension and a row variable sharing a name";
      scheme = Some "name_clash";
      run = (fun () -> ignore (name_clash () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "an empty concatenation component";
      scheme = Some "concat_empty";
      run = (fun () -> ignore (concat_empty () (input "x" [ 3 ]) : Tensor.t));
    };
    {
      name = "undetermined concatenation components, at the closure (a variable in the message)";
      scheme = None;
      run =
        (fun () ->
          let y = concat_undetermined () (input "x" [ 3 ]) in
          ignore (Shape.to_dims y.Tensor.shape : int array));
    };
    {
      name = "a failed total-elements constraint (hand-built: the solver raises it on divisibility)";
      scheme = None;
      run =
        (fun () ->
          let v = Row.get_var ~name:"k" () in
          raise
          @@ Row.Shape_error
               ( "Total_elems: the total is not divisible",
                 [
                   Row.Rows_constr_failed
                     {
                       constr =
                         Total_elems
                           { numerator = Num_elems 6; divided_by = [ v; v ]; keep_axis = false };
                     };
                 ] ));
    };
    {
      name = "the first mistake without a scheme (session ids, for contrast)";
      scheme = None;
      run = (fun () -> ignore (two_layer_unchecked ~hid:8 ~out:2 () (input "x" [ 3 ]) : Tensor.t));
    };
  ]

(* Whether a text spells a number from the reserved ranges (every one starts at 2^40). *)
let spells_reserved_id text =
  String.map text ~f:(fun c -> if Char.is_digit c then c else ' ')
  |> String.split ~on:' '
  |> List.exists ~f:(fun run ->
      (not (String.is_empty run))
      && (String.length run > 18 || Int.of_string run >= Shape.reserved_id_base))

let constraint_origins : Row.constraint_ -> Row.constraint_origin list = function
  | Dim_eq { origin; _ }
  | Dim_size { origin; _ }
  | Row_eq { origin; _ }
  | Dim_ineq { origin; _ }
  | Row_ineq { origin; _ }
  | Dim_constr { origin; _ }
  | Rows_constr { origin; _ }
  | Terminal_dim (_, _, origin)
  | Terminal_row (_, _, origin)
  | Shape_row (_, origin) ->
      origin

let error_origins trace =
  List.concat_map trace ~f:(function Row.Constraint_failed c -> constraint_origins c | _ -> [])

(* Whether an origin names a tensor of the scheme [label], judged apart from how the solver ranks
   them: the scheme itself (its result, and the origins of its application and its trace's
   solution), its trace's arguments ([label.arg<i>]), or an unlabeled tensor its trace made, whose
   id is from the reserved range. A labeled tensor the trace made counts as another's here, so this
   reading can refuse a right order, never pass a wrong one, in this file's cases. *)
let names_scheme_tensor ~label (o : Row.constraint_origin) =
  let scheme_tensor name =
    String.equal name label
    || String.is_prefix name ~prefix:(label ^ ".arg")
    || String.is_prefix name ~prefix:"n"
       && Option.exists
            (Int.of_string_opt (String.drop_prefix name 1))
            ~f:(fun id -> id >= Shape.reserved_id_base)
  in
  scheme_tensor o.lhs_name || scheme_tensor o.rhs_name

let full_sexp = Utils.get_global_flag ~default:false ~arg_name:"shape_error_full_sexp"
let cap = Int.of_string (Utils.get_global_arg ~default:"20" ~arg_name:"max_shape_error_origins")

(* The run whose command line lowers the cap below the application error's origin count. *)
let trimmed = Array.mem (Sys.get_argv ()) "trimmed" ~equal:String.equal

let () =
  let results =
    List.map cases ~f:(fun case ->
        Tensor.unsafe_reinitialize ();
        match case.run () with
        | () -> (case, None)
        | exception (Row.Shape_error (msg, trace) as exn) -> (case, Some (exn, msg, trace)))
  in
  (* Under shape_error_full_sexp the printers spell the session's raw ids: no golden of them. The
     trimmed run shows the renderings the cap trimmed, the plain run's golden the rest. *)
  List.iter results ~f:(fun (case, outcome) ->
      Stdio.printf "--- %s\n%s" case.name
        (match outcome with
        | None -> "(no error)\n"
        | Some _ when full_sexp -> ""
        | Some (_, _, trace) when trimmed && List.length (error_origins trace) < cap -> ""
        | Some (exn, _, _) -> Stdlib.Printexc.to_string exn ^ "\n"));
  let errors = List.filter_map results ~f:(fun (case, o) -> Option.map o ~f:(fun e -> (case, e))) in
  let in_scheme =
    List.filter_map errors ~f:(fun (case, e) -> Option.map case.scheme ~f:(fun label -> (label, e)))
  in
  (* The control: the raw constraint sexp of the same errors spells reserved-range ids. *)
  let raw (msg, trace) =
    Sexp.to_string (Sexp.List (Sexp.Atom msg :: List.map trace ~f:Row.sexp_of_error_trace))
  in
  let printed (exn, _, _) = Stdlib.Printexc.to_string exn in
  Verdict.case "readable shape errors" (fun () ->
      p "every case raised a shape error" (List.length errors = List.length cases);
      p_all "every application error names the scheme that refused it" in_scheme
        ~f:(fun (label, (_, msg, _)) ->
          (not (String.is_prefix msg ~prefix:"applying the shape scheme"))
          || String.is_prefix msg ~prefix:("applying the shape scheme " ^ label ^ " to"));
      p_exists "the raw constraint sexp of an error in a scheme spells a reserved-range id"
        in_scheme ~f:(fun (_, (_, msg, trace)) -> spells_reserved_id (raw (msg, trace)));
      if full_sexp then (
        (* The negative control of the readable runs' claims: the same detector, on what the
           printers show once the readable rendering is off. *)
        p_exists
          "with shape_error_full_sexp, a printed error in a scheme spells a reserved-range id"
          in_scheme ~f:(fun (_, e) -> spells_reserved_id (printed e));
        p_all "with shape_error_full_sexp, the exception's sexp is the raw one" errors
          ~f:(fun (_, (exn, msg, _)) ->
            match Exn.sexp_of_t exn with
            | Sexp.List (Atom "lib/shape.ml.Shape_error" :: Atom m :: _) -> String.equal m msg
            | _ -> false))
      else (
        p_none "no rendering spells a reserved-range id" errors ~f:(fun (_, e) ->
            spells_reserved_id (printed e));
        p_all "every rendering fits one screen (at most 24 lines)" errors ~f:(fun (_, e) ->
            List.length (String.split_lines (printed e)) <= 24);
        (* [Exn.to_string] (the sexp converter) carries the same lines, one atom each. *)
        p_all "the exception's sexp carries the rendering's lines" errors
          ~f:(fun (_, (exn, msg, trace)) ->
            Sexp.equal (Exn.sexp_of_t exn)
              (Sexp.List
                 (Sexp.Atom "Shape_error"
                 :: List.map
                      (String.split_lines (Shape.error_to_string_hum msg trace))
                      ~f:(fun l -> Sexp.Atom (String.strip l))))));
      (* gh-ocannl-1903: the origins naming the failing scheme's tensors come first. *)
      let scheme_flags label trace =
        List.map (error_origins trace) ~f:(names_scheme_tensor ~label)
      in
      p_all "every scheme error lists the origins naming the scheme's tensors first" in_scheme
        ~f:(fun (label, (_, _, trace)) ->
          List.is_sorted (scheme_flags label trace) ~compare:(fun a b -> Bool.compare b a));
      p_all "no error holds more origins than the cap" errors ~f:(fun (_, (_, _, trace)) ->
          List.length (error_origins trace) <= cap);
      if trimmed then
        (* The untrimmed run shows the application error's 7 origins, 4 of them the scheme's. *)
        p_exists "the cap trimmed some scheme error's origins to the scheme's own" in_scheme
          ~f:(fun (label, (_, _, trace)) ->
            let flags = scheme_flags label trace in
            cap > 0 && List.length flags = cap && List.count flags ~f:Fn.id = cap)
      else
        p_exists "some scheme error's origins name both the scheme's tensors and others" in_scheme
          ~f:(fun (label, (_, _, trace)) ->
            let flags = scheme_flags label trace in
            List.exists flags ~f:Fn.id && List.exists flags ~f:not))
