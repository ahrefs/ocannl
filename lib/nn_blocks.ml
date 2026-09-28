(** {1 Neural Network Building Blocks}

    This file contains basic building blocks for neural networks, with limited functionality. Feel
    free to copy-paste and modify as needed.

    Design principles, OCANNL fundamentals, and common patterns:
    - "Principle of least commitment": use row variables where axis count doesn't matter
    - Einsum specs here often use single-char mode (no commas) but with spaces for readability
    - Pooling uses constant kernels (stretch 0.0) to propagate window dimensions
    - conv2d uses convolution syntax: "stride*out+kernel," (often in multi-char mode)
    - Input axes (before →) for kernels show intent (and end up rightmost for memory locality)
    - Inline params \{ \} are always learnable and are lifted to unit parameter ()
    - Introduce inputs to a block after sub-block construction (sub-blocks have no automatic lifting
      like there is for inline definitions of params)
    - Always use literal strings with einsum operators when capturing variables
    - Avoid unnecessary variable captures in einsum operators, be mindful they can shadow other
      identifiers *)

open! Base
open Ocannl_tensor.Operation.DSL_modules
module Tn = Ir.Tnode

let%op box_muller grad_spec init_f () =
  let epsilon = [%oc Float.ldexp 1. (-24)] in
  let one_minus_epsilon = [%oc 1. -. Float.ldexp 1. (-24)] in
  let u1 = init_f () in
  let u2 = init_f () in
  (* The clamp away from 0 is written as [epsilon + (1 - epsilon) * u1] so that [u1] appears once,
     under scalar-operand ops only, and the final pointmul carries an equality spec: pointwise
     broadcasting would otherwise let the uniform draws close to smaller shapes than the result and
     repeat random values along the broadcast axes. *)
  Ocannl_tensor.Operation.pointmul ~grad_spec ~spec:"..b..; ..b.. => ..b.."
    (sqrt (-2. *. log (!.epsilon + (!.one_minus_epsilon *. u1))))
    (cos (2. *. !.Float.pi *. u2))

let kaiming_impl ?(scale_sq = 6.0) grad_spec init_f () =
  let w_raw = init_f () in
  let%op _ = w_raw ++ "...|..i.. -> ... => |->0" [ "i" ] in
  Ocannl_tensor.Operation.pointmul ~grad_spec w_raw [%op sqrt (!.scale_sq /. dim i)]

let xavier_impl ?(scale_sq = 6.0) grad_spec init_f () =
  let w_raw = init_f () in
  let%op _ = w_raw ++ "...|..i.. -> ..o.. => |->0" [ "i"; "o" ] in
  Ocannl_tensor.Operation.pointmul ~grad_spec w_raw [%op sqrt (!.scale_sq /. (dim i + dim o))]

[%%extend_dsls
let normal () = [%oc box_muller grad_spec uniform ()]
let normal1 () = [%oc box_muller grad_spec uniform1 ()]
let normal_at counter = [%oc box_muller grad_spec (fun () -> uniform_at counter) ()]
let normal_at1 counter = [%oc box_muller grad_spec (fun () -> uniform_at1 counter) ()]
let kaiming ?scale_sq init_f () = [%oc kaiming_impl ?scale_sq grad_spec init_f ()]
let xavier ?scale_sq init_f () = [%oc xavier_impl ?scale_sq grad_spec init_f ()]

let kaiming_at ?scale_sq init_f counter =
  [%oc kaiming_impl ?scale_sq grad_spec (fun () -> init_f counter) ()]

let xavier_at ?scale_sq init_f counter =
  [%oc xavier_impl ?scale_sq grad_spec (fun () -> init_f counter) ()]]

open DSL_modules

(* Bits-preserving conversion: ids in [2^31, 2^32) are valid uint32 values whose int32
   representation is negative, so [of_int_exn] would wrongly reject them. *)

(** Convert a list of integers to a compact tensor of class IDs (no [num_classes] allocation).
    @param lst List of integer class indices (0-based)
    @return
      A tensor of shape [len] (a [len]-sized batch axis) holding the IDs in uint32 precision, so
      that the gh-343 embedding gather can guard the dynamic index with native integer comparisons
      (no double-precision guard, no integrality check). *)
let set_uint32_id ~fn_name genarray idx id =
  if id < 0 || id > 0xFFFF_FFFF then
    invalid_arg [%string "%{fn_name}: id %{id#Int} is out of the uint32 range"];
  Bigarray.Genarray.set genarray idx (Int32.of_int_trunc id)

let class_ids_of_int_list ?(label = "class_ids") lst =
  let open Bigarray in
  let arr = lst |> Array.of_list in
  let len = Array.length arr in
  let genarray = Genarray.create Int32 c_layout [| len |] in
  for i = 0 to len - 1 do
    set_uint32_id ~fn_name:"class_ids_of_int_list" genarray [| i |] arr.(i)
  done;
  TDSL.rebatch ~l:label (Ir.Ndarray.as_array Ir.Ops.Uint32 genarray) ()

(** Build a logical one-hot tensor from a tensor of class IDs, using only existing operations
    ([range] + equality) so the compiler keeps the proof that the result is one-hot (enabling the
    gh-343 embedding gather optimization). No dense [len * num_classes] data is materialized on the
    host. With [ids] shaped as a [len] batch (output rank 0), the result is [len; num_classes]:
    [one_hot[i, k] = (k == ids[i])].
    @param num_classes The number of classes (size of the one-hot dimension). *)
let one_hot_of_ids ~num_classes ids =
  (* Class IDs are integer-valued: flow an integer precision backward into [ids], like the threefry
     operations do with uint4x32, so the gather guard runs in native integer comparisons. This is
     soft: a tensor with an explicitly specified or already-settled precision (e.g. float IDs) is
     unaffected and keeps the double-precision guard path. *)
  if not (Lazy.is_val ids.Tensor.value.Tn.storage_prec) then
    Tn.update_infer_prec ids.Tensor.value (lazy Ir.Ops.uint32);
  let classes = TDSL.range num_classes in
  let open TDSL.O in
  classes = ids

(** Convert a list of integers to a logical one-hot encoded tensor of shape [len; num_classes]. This
    composes {!class_ids_of_int_list} and {!one_hot_of_ids}: it stores only [len] compact IDs on the
    host and expresses the one-hot logically, rather than allocating a dense [len * num_classes]
    Bigarray. See {!dense_one_hot_of_int_list} if a materialized host one-hot is genuinely required.
    @param num_classes The number of classes (size of the one-hot dimension)
    @param lst List of integer class indices (0-based) *)
let one_hot_of_int_list ~num_classes lst = one_hot_of_ids ~num_classes (class_ids_of_int_list lst)

(** Convert a list of integers to a dense, host-materialized one-hot Bigarray-backed tensor of shape
    [len; num_classes]. Prefer {!one_hot_of_int_list} (logical) unless a dense host fixture is
    needed; a materialized Bigarray carries no proof that it is one-hot, so it cannot be optimized
    into an embedding gather. *)
let dense_one_hot_of_int_list ~num_classes lst =
  let open Bigarray in
  let len = List.length lst in
  let arr = lst |> Array.of_list in
  let genarray = Genarray.create Float32 c_layout [| len; num_classes |] in
  for i = 0 to len - 1 do
    for j = 0 to num_classes - 1 do
      Genarray.set genarray [| i; j |] 0.
    done
  done;
  for i = 0 to len - 1 do
    Genarray.set genarray [| i; arr.(i) |] 1.
  done;
  TDSL.rebatch ~l:"one_hot" (Ir.Ndarray.as_array Ir.Ops.Single genarray) ()

(** Convert an array of token IDs (e.g. the output of [Dataprep.Bpe.encode]) to a tensor of shape
    [len] (a [len]-sized batch axis, output rank 0) holding the IDs in uint32 precision -- large
    enough for any practical vocabulary (e.g. Gemma's 256K) and integer-native for the gh-343
    embedding gather (see {!class_ids_of_int_list}). Feed the result to {!one_hot_of_ids} for an
    embedding lookup.

    If [max_len] is given, the sequence is truncated or right-padded with [pad_id] to exactly
    [max_len] (needed for fixed-shape batched inference). Prepending/appending special tokens
    (BOS/EOS) is the caller's responsibility.
    @param pad_id The token ID used for padding (default 0)
    @param max_len If given, the result has exactly this length regardless of [Array.length ids] *)
let token_ids_of_array ?(label = "token_ids") ?max_len ?(pad_id = 0) ids =
  let open Bigarray in
  let len = Option.value max_len ~default:(Array.length ids) in
  let genarray = Genarray.create Int32 c_layout [| len |] in
  for i = 0 to len - 1 do
    let id = if i < Array.length ids then ids.(i) else pad_id in
    set_uint32_id ~fn_name:"token_ids_of_array" genarray [| i |] id
  done;
  (* Not [rebatch]: a Reshape-inferred batch row only constrains the total element count, so a
     length-1 sequence would collapse to a scalar instead of a [1] batch. *)
  TDSL.wrap ~l:label ~b:[ len ] ~o:[] (Ir.Ndarray.as_array Ir.Ops.Uint32 genarray) ()

(** Batched variant of {!token_ids_of_array}: convert several token-ID sequences to a single tensor
    of shape [num_seqs; max_len] (two batch axes, output rank 0) in uint32 precision. Each sequence
    is truncated or right-padded with [pad_id] to [max_len], which defaults to the length of the
    longest sequence. Composes with {!one_hot_of_ids} the same way as the unbatched variant, giving
    a [num_seqs; max_len; num_classes] logical one-hot.
    @param pad_id The token ID used for padding (default 0)
    @param max_len The common sequence length (default: longest sequence in [seqs]) *)
let token_ids_of_batch ?(label = "token_ids") ?max_len ?(pad_id = 0) seqs =
  let open Bigarray in
  let num_seqs = Array.length seqs in
  if num_seqs = 0 then invalid_arg "token_ids_of_batch: the batch must be non-empty";
  let max_len =
    match max_len with
    | Some len -> len
    | None -> Array.fold seqs ~init:0 ~f:(fun acc s -> max acc (Array.length s))
  in
  let genarray = Genarray.create Int32 c_layout [| num_seqs; max_len |] in
  for s = 0 to num_seqs - 1 do
    let seq = seqs.(s) in
    for i = 0 to max_len - 1 do
      let id = if i < Array.length seq then seq.(i) else pad_id in
      set_uint32_id ~fn_name:"token_ids_of_batch" genarray [| s; i |] id
    done
  done;
  TDSL.wrap ~l:label ~b:[ num_seqs; max_len ] ~o:[] (Ir.Ndarray.as_array Ir.Ops.Uint32 genarray) ()

(** Gaussian Error Linear Unit, tanh approximation (Hendrycks & Gimpel 2016):
    [0.5 * x * (1 + tanh (sqrt (2/pi) * (x + 0.044715 * x^3)))]. This is the GPT-2 activation (HF
    [gelu_new]); it uses only existing primitives (notably [Tanh_approx]), no dedicated backend
    support needed. *)
let%op gelu x = 0.5 *. x *. (1.0 + tanh (0.7978845608028654 *. (x + (0.044715 *. (x *. x *. x)))))

let%op mlp_layer ~label ~hid_dim () x = relu (({ w } * x) + { b = 0.; o = [ hid_dim ] })

(** Masks and scales by 1/keep_prob to maintain expected value. When [train_step = None], the
    dropout rate is ignored and the tensor is returned unmodified. *)
let%op dropout ~rate () ~train_step x =
  match train_step with
  | Some train_step when Float.(rate > 0.0) ->
      x *. (!.rate < uniform_at !@train_step) /. (1.0 - !.rate)
  | _ -> x

(** Multi-layer perceptron of depth [List.length hid_dims + 1], with a linear output layer. *)
let%op mlp ~label ~hid_dims () =
  let layers =
    List.mapi hid_dims ~f:(fun i hid_dim ->
        mlp_layer ~label:(("L" ^ Int.to_string i) :: label) ~hid_dim ())
  in
  fun x ->
    let hidden = List.fold layers ~init:x ~f:(fun x layer -> layer x) in
    { w_out } * hidden

let reduce_specified_axes spec =
  let lhs =
    if String.contains spec ',' then
      Str.global_replace (Str.regexp "[A-Za-z][A-Za-z_0-9]*") "0" spec
    else Str.global_replace (Str.regexp "[A-Za-z]") "0" spec
  in
  spec ^ " => " ^ lhs

(** Softmax across specified axes. Does not support non-default row variables.

    The row max is taken through the non-differentiable DSL: softmax is invariant under a shift of
    its input, so the gradient through the max is exactly zero, and the composed autodiff would
    otherwise spend a per-row reduction and an argmax-equality pass computing that zero up to
    rounding -- and inject the rounding at the argmax cells. Applying an [NTDSL] operation to a
    differentiable operand cuts the gradient at that operation ([Prohibit_grad]); [stop_gradient] is
    the identity instance of the same mechanism, at the price of a copy. *)
let%op softmax ~spec ?(temperature = 1.0) () =
  let spec = reduce_specified_axes spec in
  fun x ->
    let x_scaled = if Float.(temperature <> 1.0) then x /. !.temperature else x in
    let max_vals = NTDSL.O.(x_scaled @^^ spec) in
    let exp_vals = exp (x_scaled - max_vals) in
    exp_vals /. (exp_vals ++ spec)

(** Cross-entropy loss from raw logits, adopting the llm.c [fused_classifier] contract (issue
    gh-464): numerically stable log-sum-exp (never [log (softmax logits)]), written so that the
    softmax-probabilities intermediate stays virtual — never materialized in either pass — and the
    backward accumulates [probs - targets] directly into the logits gradient. The row-wise max is
    taken through the non-differentiable DSL, as in {!softmax}: the max term cancels exactly in the
    loss, so no gradient should flow through it (llm.c likewise treats it as a constant).

    [spec] follows the {!softmax} convention: it names the class/vocabulary axes to reduce over
    (e.g. ["... | v"]). [targets] is a probability distribution over those axes (typically one-hot;
    label smoothing works too). [mask] multiplies the per-position losses (e.g. 0/1 to exclude
    padding tokens); [normalize_by] divides the summed loss (e.g. by a valid-token count). Without
    [mask] and [normalize_by], returns the summed scalar loss. *)
let%op cross_entropy_loss ~spec ?mask ?normalize_by () ~logits ~targets =
  let reduce_spec = reduce_specified_axes spec in
  let max_logits = NTDSL.O.(logits @^^ reduce_spec) in
  let shifted = logits - max_logits in
  let log_probs = shifted - log (exp shifted ++ reduce_spec) in
  let nll = neg ((targets *. log_probs) ++ reduce_spec) in
  let masked_nll = match mask with None -> nll | Some m -> nll *. m in
  (* Reduce all three axis kinds: [spec] may name class axes in the input row (e.g. the attention
     convention "... | t -> ..."), in which case [nll] still has an input row here. *)
  let cross_entropy = masked_nll ++ "...|...->... => |->0" in
  match normalize_by with None -> cross_entropy | Some n -> cross_entropy /. n

(** {2 Position Embedding Strategies} *)

(** Strategy for positional encoding in attention / transformer blocks. *)
type position_embedding =
  | Learned_additive  (** Current default: learned parameter added to input embeddings. *)
  | Sinusoidal_additive of { enc_encoding : Tensor.t; dec_encoding : Tensor.t }
      (** Fixed sinusoidal encoding added to input embeddings. Use separate tensors for encoder and
          decoder when [d_enc <> d_dec]. For equal widths, the same tensor can be passed for both.
          Build with {!sinusoidal_position_encoding}. *)
  | RoPE of { freqs : Tensor.t; positions : Tensor.t }
      (** Rotary embeddings applied to Q/K inside self-attention. No additive component. *)
  | No_pos_embed  (** No position information. *)

(* PoPE (arXiv:2509.10534) deferred to #444. PoPE maps d scalars → 2d reals (softplus magnitude ×
   position phase), doubling dimensionality. Requires decoupling projection width from head width in
   multi_head_attention. See #444. *)

(** RoPE inverse frequencies: theta_k = base^(-2k/d) for k = 0..half_d-1.
    @param half_d Per-head key dimension divided by 2 (i.e. d_k / 2, NOT d_model / 2).
    @param base Default 10000.0; some models use 500000 for long contexts. *)
let rope_frequencies ~half_d ?(base = 10000.0) () =
  NTDSL.init ~l:"rope_freqs" ~prec:Ir.Ops.single ~b:[] ~i:[] ~o:[ half_d ]
    ~f:(function
      | [| k |] -> Float.(base ** (of_int Int.(-2 * k) /. of_int Int.(2 * half_d)))
      | _ -> assert false)
    ()

(** Position indices [0, 1, ..., seq_len-1] as a non-learned batch-dim tensor. *)
let position_indices ~seq_len () =
  NTDSL.init ~l:"pos_idx" ~prec:Ir.Ops.single ~b:[ seq_len ] ~i:[] ~o:[]
    ~f:(function [| pos |] -> Float.of_int pos | _ -> assert false)
    ()

(** Sinusoidal positional encoding (Vaswani et al. 2017). Non-learned, shape: batch_dims=[max_len],
    output_dims=[d_model]. Matches model width at the transformer input level, NOT per-head width.
*)
let sinusoidal_position_encoding ~d_model ~max_len () =
  NTDSL.init ~l:"sinusoidal_pe" ~prec:Ir.Ops.single ~b:[ max_len ] ~i:[] ~o:[ d_model ]
    ~f:(function
      | [| pos; i |] ->
          let i_even = i - (i % 2) in
          let div_term = Float.(10000. ** (of_int i_even /. of_int d_model)) in
          let angle = Float.of_int pos /. div_term in
          if i % 2 = 0 then Float.sin angle else Float.cos angle
      | _ -> assert false)
    ()

(** Apply RoPE rotation to tensor [x] whose last output axis has even size [d]. Rotates within the
    last output axis (per-head width [d]) without crossing head boundaries. [freqs] has
    output=[d/2], [positions] has batch=[seq_len]. *)
let%op rope ~freqs ~positions x =
  (* Compute angles (pos * theta_k) separately for cos/sin to avoid a framework issue where a shared
     intermediate tensor loses its computation table entry. *)
  let cos_a = cos (positions *. freqs) in
  let sin_a = sin (positions *. freqs) in
  (* Split last output axis into even/odd pairs *)
  let x_even = deinterleave_even x in
  let x_odd = deinterleave_odd x in
  (* Pairwise rotation: out_even = x_even * cos(angle) - x_odd * sin(angle) out_odd = x_even *
     sin(angle) + x_odd * cos(angle) *)
  let out_even = (x_even *. cos_a) - (x_odd *. sin_a) in
  let out_odd = (x_even *. sin_a) + (x_odd *. cos_a) in
  interleave out_even out_odd

(** The score a masked-out attention position is filled with before the softmax:
    [Float.neg_infinity], so that [exp (fill - max)] is exactly zero.

    This is deliberately not a large finite magic number. A finite sentinel has to be chosen against
    a storage precision — the long-standing [-1e9] is four orders of magnitude past fp16's largest
    finite value, so it tripped [Ops.exceeds_fp16_cutoff] and made every masked attention
    unlowerable at half precision. Negative infinity is exactly representable in every float format
    OCANNL supports, and says what is meant: this position is below anything the format can hold.

    The one behavior a finite fill bought is that a {e fully} masked row (no live position at all)
    degrades to a uniform distribution rather than to NaN — with an all-[-inf] row the softmax's
    max-subtraction yields [-inf - -inf = nan]. Causal masks always leave the diagonal live and are
    unaffected; a padding mask that can mask a whole row should pass a finite [?mask_fill] (e.g.
    [-1e4], which still underflows [exp] to zero at every precision). *)
let default_mask_fill = Float.neg_infinity

let%op multi_head_attention ~label ~num_heads ~d_k ~d_v ?temperature ?(dropout_rate = 0.0)
    ?(mask_fill = default_mask_fill) ?(pos_embed = No_pos_embed) () ~train_step ?mask x =
  (match pos_embed with RoPE _ -> assert (Int.(d_k % 2 = 0)) | _ -> ());
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  (* RoPE rotates within the last output axis (d, the per-head width). The h axis (heads) is
     preserved — no rotation across head boundaries. *)
  let q, k =
    match pos_embed with
    | RoPE { freqs; positions } -> (rope ~freqs ~positions q, rope ~freqs ~positions k)
    | _ -> (q, k)
  in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  (* NOTE: often d_k = d_v = d_model / num_heads, but we allow for other values. *)
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  (* We don't need to lift [softmax ~spec ()] because it doesn't introduce any new params. *)
  let attn_weights =
    softmax ~spec:" ... | t -> ..." ?temperature ()
      (match mask with None -> scores | Some mask -> where mask scores !.mask_fill)
  in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  (* w_o output shape will automatically be set to the model dimension(s) by shape inference. *)
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

let%op multi_head_att_workshop ~num_heads ~d_k ~d_v () x =
  let q = { w_q } * x in
  let k = { w_k } * x in
  let v = { w_v } * x in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h" [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights = softmax ~spec:" ... | t -> ..." () scores in
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

let%op layer_norm ~label ?(epsilon = 1e-5) () x =
  let mean = (x ++ " ... | ..d..  => ... | 0 " [ "d" ]) /. dim d in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ " ... | ... => ... |  0 ") /. dim d in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  (* gamma and beta are learned, but initialized to good defaults *)
  ({ gamma = 1. } *. normalized) + { beta = 0. }

let%op transformer_encoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5)
    ?(pos_embed = No_pos_embed) () =
  let mha = multi_head_attention ~label:("mha" :: label) ~num_heads ~d_k ~d_v ~pos_embed () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step input ->
    let x1 = ln1 (input + mha ~train_step input) in
    ln2 (x1 + ffn x1)

(** Decoder-only transformer block: masked self-attention + FFN with post-norm LayerNorm. Like
    {!transformer_encoder_block} but accepts a [~mask] parameter for causal masking. No
    cross-attention — suitable for autoregressive language models. *)
let%op decoder_only_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) ?(dropout_rate = 0.0)
    ?mask_fill ?(pos_embed = No_pos_embed) () =
  let masked_mha =
    multi_head_attention ~label:("masked_mha" :: label) ~num_heads ~d_k ~d_v ~dropout_rate
      ?mask_fill ~pos_embed ()
  in
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  fun ~train_step x ~mask ->
    let x1 = ln1 (x + masked_mha ~train_step ~mask x) in
    ln2 (x1 + ffn x1)

(** Stack of {!decoder_only_block} layers. *)
let decoder_only ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff ?epsilon ?dropout_rate ?mask_fill
    ?(pos_embed = No_pos_embed) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        decoder_only_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k ~d_v ~d_ff ?epsilon ?dropout_rate ?mask_fill ~pos_embed ())
  in
  fun ~train_step x ~mask -> List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step x ~mask)

(* Cross-attention does not apply RoPE — position encoding is for self-attention only. *)
let%op cross_attention ~label ~num_heads ~d_k ~d_v ?temperature ?(dropout_rate = 0.0) () ~train_step
    x ~enc_output =
  let q = { w_q } * x in
  let k = { w_k } * enc_output in
  let v = { w_v } * enc_output in
  let scores =
    (q +* k " ... s | h d; ... t | h d => ... s | t -> h " [ "h"; "d" ]) /. sqrt (dim d)
  in
  Shape.set_dim h num_heads;
  Shape.set_dim d d_k;
  Shape.set_dim e d_v;
  let attn_weights = softmax ~spec:" ... | t -> ..." ?temperature () scores in
  let attn_weights = dropout ~rate:dropout_rate () ~train_step attn_weights in
  { w_o } * (attn_weights +* v " ... s | t -> h; ... t | h e => ... s | h e" [ "e" ])

let%op transformer_decoder_block ~label ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) ?mask_fill
    ?(pos_embed = No_pos_embed) () =
  (* RoPE is applied to self-attention only, not cross-attention. *)
  let masked_mha =
    multi_head_attention ~label:("masked_mha" :: label) ~num_heads ~d_k ~d_v ?mask_fill ~pos_embed
      ()
  in
  let cross_mha = cross_attention ~label:("cross_mha" :: label) ~num_heads ~d_k ~d_v () in
  (* Standard 2-layer FFN: expand to d_ff then contract back to d_model *)
  let ffn = mlp ~label:("ffn" :: label) ~hid_dims:[ d_ff ] () in
  let ln1 = layer_norm ~label:("ln1" :: label) ~epsilon () in
  let ln2 = layer_norm ~label:("ln2" :: label) ~epsilon () in
  let ln3 = layer_norm ~label:("ln3" :: label) ~epsilon () in
  fun ~train_step target ~enc_output ~mask ->
    let x1 = ln1 (target + masked_mha ~train_step ~mask target) in
    let x2 = ln2 (x1 + cross_mha ~train_step x1 ~enc_output) in
    ln3 (x2 + ffn x2)

let transformer_encoder ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5)
    ?(pos_embed = No_pos_embed) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_encoder_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k ~d_v ~d_ff ~epsilon ~pos_embed ())
  in
  fun ~train_step x -> List.fold layers ~init:x ~f:(fun x layer -> layer ~train_step x)

let transformer_decoder ~label ~num_layers ~num_heads ~d_k ~d_v ~d_ff ?(epsilon = 1e-5) ?mask_fill
    ?(pos_embed = No_pos_embed) () =
  let layers =
    List.init num_layers ~f:(fun i ->
        transformer_decoder_block
          ~label:(("layer" ^ Int.to_string i) :: label)
          ~num_heads ~d_k ~d_v ~d_ff ~epsilon ?mask_fill ~pos_embed ())
  in
  fun ~train_step target ~enc_output ~mask ->
    List.fold layers ~init:target ~f:(fun x layer -> layer ~train_step x ~enc_output ~mask)

let%op transformer ~label ~num_encoder_layers ~num_decoder_layers ~num_heads ~d_enc ~d_dec ~d_ff
    ?(epsilon = 1e-5) ?mask_fill ?(pos_embed = Learned_additive) () =
  let enc_att = [%oc d_enc / num_heads] in
  let dec_att = [%oc d_dec / num_heads] in
  let attn_pos_embed = match pos_embed with RoPE _ as pe -> pe | _ -> No_pos_embed in
  let encoder =
    transformer_encoder ~label:("encoder" :: label) ~num_layers:num_encoder_layers ~num_heads
      ~d_k:enc_att ~d_v:enc_att ~d_ff ~epsilon ~pos_embed:attn_pos_embed ()
  in
  let decoder =
    transformer_decoder ~label:("decoder" :: label) ~num_layers:num_decoder_layers ~num_heads
      ~d_k:dec_att ~d_v:dec_att ~d_ff ~epsilon ?mask_fill ~pos_embed:attn_pos_embed ()
  in
  (* NOTE: { pos_encoding } and { pos_encoding_tgt } are learned inline params lifted by %op. They
     are created unconditionally but only used when pos_embed = Learned_additive. *)
  let pos_encoding_tgt = if Int.(d_enc = d_dec) then pos_encoding else { pos_encoding_tgt } in
  fun ~train_step ~src ~tgt ~mask ->
    let enc_input = { src_embed; o = [ d_enc ] } * src in
    let enc_output =
      match pos_embed with
      | Learned_additive -> encoder ~train_step (enc_input + { pos_encoding })
      | Sinusoidal_additive { enc_encoding; _ } -> encoder ~train_step (enc_input + enc_encoding)
      | RoPE _ | No_pos_embed -> encoder ~train_step enc_input
    in
    let tgt_embedded_base = { tgt_embed; o = [ d_dec ] } * tgt in
    let tgt_embedded =
      match pos_embed with
      | Learned_additive -> tgt_embedded_base + pos_encoding_tgt
      | Sinusoidal_additive { dec_encoding; _ } -> tgt_embedded_base + dec_encoding
      | RoPE _ | No_pos_embed -> tgt_embedded_base
    in
    { w_out } * decoder ~train_step tgt_embedded ~enc_output ~mask

(** Transformer with teacher forcing for autoregressive training.

    TODO: Simplify once tensor shifting/slicing is better supported in shape inference. Currently
    requires pre-shifted tgt_input (all but last token) and tgt_target (all but first token). During
    training, the model learns to predict tgt_target given tgt_input. *)
let%op transformer_with_loss ~label:_ ~model () ~train_step ~src ~tgt_input ~tgt_target ~mask =
  (* Get model predictions for the input sequence *)
  let logits = model ~train_step ~src ~tgt:tgt_input ~mask in

  (* Numerically stable cross-entropy over the vocabulary dimension; tgt_target should be one-hot
     encoded or use label smoothing *)
  let loss = cross_entropy_loss ~spec:"... | v" () ~logits ~targets:tgt_target in

  (* Return both loss and logits for potential additional metrics *)
  (loss, logits)

(** {2 Convolutional Neural Network Building Blocks} *)

(** 2D convolution layer with flexible padding and stride options.

    When [use_padding=false] and [stride > 1], the input spatial dimensions must satisfy:
    [(input_size - kernel_size) mod stride = 0], otherwise shape inference will fail with
    "incompatible stride" error. The output size is [(input_size - kernel_size) / stride + 1].

    When [use_padding=true], there is no such restriction and output size is [input_size / stride].

    @param out_channels
      Optional number of output channels. If not provided, must be inferred from context (e.g., from
      a downstream operation that constrains the output shape). *)
let%op conv2d ~label ?(kernel_size = 3) ?(stride = 1) ?(use_padding = true) ?out_channels () x =
  (* Notation: kernel height (kh), kernel width (kw), input channels (ic), output channels (oc),
     output height (oh), output width (ow) *)
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  Option.iter out_channels ~f:(Shape.set_dim oc);
  x
  +* { kernel }
       "... | stride*oh+kh, stride*ow+kw, ..ic..; |kh, kw, ..ic.. -> ..oc.. => ... | oh, ow, ..oc.."
       [ "kh"; "kw"; "oc" ]
  (* The spec'd add keeps the bias per-channel: a plain [+] would broadcast-unify the bias to the
     full feature map [oh, ow, ..oc..]. *)
  +++ { bias = 0. } "... | oh, ow, ..oc..; |..oc.. => ... | oh, ow, ..oc.."

(** Depthwise separable convolution - more efficient for mobile/edge devices. Consists of depthwise
    conv (spatial filtering per channel) followed by pointwise conv (1x1 conv for channel mixing).

    See {!conv2d} for dimension constraints when [use_padding=false]. *)
let%op depthwise_separable_conv2d ~label ?(kernel_size = 3) ?(stride = 1) ?(use_padding = true) () x
    =
  (* Depthwise: each input channel is convolved with its own filter *)
  Shape.set_dim kh kernel_size;
  Shape.set_dim kw kernel_size;
  let depthwise =
    x
    +* { dw_kernel }
         "... | stride*oh+kh, stride*ow+kw, ..ic..; |kh, kw -> ..ic.. => ... | oh, ow, ..ic.."
         [ "kh"; "kw" ]
  in
  (* Pointwise: 1x1 conv to mix channels *)
  depthwise
  +* { pw_kernel } "... | h, w, ..ic..; |..ic.. -> ..oc.. => ... | h, w, ..oc.."
  +++ { bias = 0. } "... | h, w, ..oc..; |..oc.. => ... | h, w, ..oc.."

(** Max pooling for 2D spatial data - reduces spatial dimensions by taking maximum values.

    Without padding ([use_padding=false], the default), the input spatial dimensions must satisfy:
    [(input_size - window_size) mod stride = 0], otherwise shape inference will fail. The output
    size is [(input_size - window_size) / stride + 1]. The [<] in the einsum spec indicates
    no-padding mode (indices stay within bounds).

    With [use_padding=true] ("same" pooling, output size [input_size / stride]), the pool lowers
    with clamped window bounds (gh-504): per output position the window loop is range-guarded to the
    intersection of the window with the valid input range, and an out-of-range position contributes
    the max-accumulation identity ([-inf]) — the same as not visiting it — so clamping is
    semantically exact and the operand needs NO margins at all. The operand therefore composes
    freely with 0-neutral margin-touching consumers such as padded convs (the Inception-block
    pattern), with eagerly allocated data nodes (e.g. [TDSL.init]), and with operands already
    lowered in earlier compilations of a staged flow. The backward argmax scatter transposes the
    clamp (guarded writes). Interior outputs still see full windows: the guards flip truth only at
    affine breakpoints of the output axis, so [Schedule.Partition] at
    [Schedule.partition_breakpoints] specializes guard-free interior segments (gh-508).

    Overlapping pooling ([stride < window_size], AlexNet-style) has exact gradients: the gradient
    gate lives in the (output x window) product space (gh-512), so each position receives gradient
    from exactly the windows it won, with ties gating every achieving pair. Non-overlapping pooling
    ([stride >= window_size], the common case) dispatches to the cheaper input-space gate (gh-527) —
    exact on that domain, ties included, see [Operation.tropical]. *)
let%op max_pool2d ?(stride = 2) ?(window_size = 2) ?(use_padding = false) () x =
  (* [@^+] expands to the [tropical] in scope (TDSL.O's, unless shadowed) — dispatch the gradient
     gate here, where the window geometry is a plain value. *)
  let tropical ?label ?capture_dims spec t1 t2 =
    tropical ?label ?capture_dims ~nonoverlapping:(Int.( >= ) stride window_size) spec t1 t2
  in
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  (* NOTE: projections inference runs per-assignment in a distinct phase from shape inference, so
     for it to know about the window size, we use a constant kernel = 0.0 to propagate the shape.
     [stretch] makes the kernel's shape resolve at this use site — it acquires the window axes from
     the einsum spec (gh-ocannl-544; plain operation results close down to their arguments' shapes).
     See: https://github.com/ahrefs/ocannl/discussions/381 *)
  Shape.set_dim pwh window_size;
  Shape.set_dim pww window_size;
  if use_padding then
    x
    @^+ "... | stride*oh= + pwh, stride*ow= + pww, ..c..; |pwh, pww => ... | oh, ow, ..c.."
          [ "pwh"; "pww" ] (stretch 0.0)
  else
    x
    @^+ "... | stride*oh< + wh, stride*ow< + ww, ..c..; |wh, ww => ... | oh, ow, ..c.."
          [ "wh"; "ww" ] (stretch 0.0)

(** Like {!max_pool2d}, but with [use_padding=true] the pool reads a private materialized copy of
    the operand instead of the operand itself. Since the clamped-window lowering (gh-504) removed
    the pool's [-inf] margin demand, {!max_pool2d} composes with any operand and the copy is no
    longer needed as a conflict remedy — this variant is kept as the materialized-copy remedy
    pattern for the remaining conflict class (margin-touching consumers with different FINITE
    neutral elements, e.g. margins committed by [wrap_padded] to a value a padded conv's 0 neutral
    conflicts with), and for explicitly decoupling the pool's read from a shared buffer.

    Prefer {!max_pool2d}: it reads the operand directly, without the extra buffer and copy. *)
let%op max_pool2d_copy ?(stride = 2) ?(window_size = 2) ?(use_padding = false) () x =
  (* Same gradient-gate dispatch as {!max_pool2d} (gh-527). *)
  let tropical ?label ?capture_dims spec t1 t2 =
    tropical ?label ?capture_dims ~nonoverlapping:(Int.( >= ) stride window_size) spec t1 t2
  in
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  Shape.set_dim pwh window_size;
  Shape.set_dim pww window_size;
  if use_padding then
    (* Note: the copy's beg-anchored row composes with beg-anchored producers (einsum outputs like
       conv2d's [oh, ow, ..oc..]), closed rows (data nodes), and — since the mixed-anchoring stage-6
       closing in [Row.unify_row] — open trailing-dims rows (plain broadcast results). *)
    let x_pool = x ++ "... | h, w, ..c.. => ... | h, w, ..c.." in
    x_pool
    @^+ "... | stride*oh= + pwh, stride*ow= + pww, ..c..; |pwh, pww => ... | oh, ow, ..c.."
          [ "pwh"; "pww" ] (stretch 0.0)
  else
    x
    @^+ "... | stride*oh< + wh, stride*ow< + ww, ..c..; |wh, ww => ... | oh, ow, ..c.."
          [ "wh"; "ww" ] (stretch 0.0)

(** Average pooling for 2D spatial data - reduces spatial dimensions by averaging values.

    See {!max_pool2d} for dimension constraints. Note there is no [use_padding] option: a padded
    ([=]-mode) add-family window keeps the physical 0-margins mechanism (finite accumulation
    identity — clamped lowering, gh-504, applies only to the max/tropical family), which gives
    [count_include_pad] semantics — the divisor is the full window size everywhere, counting the
    zero margins. PyTorch's default is [count_exclude_pad] (per-position valid-window divisor);
    adding a padded average pool is a per-op semantic decision that should document its choice. *)
let%op avg_pool2d ?(stride = 2) ?(window_size = 2) () x =
  Shape.set_dim wh window_size;
  Shape.set_dim ww window_size;
  let sum =
    x
    +++ "... | stride*oh< + wh, stride*ow< + ww, ..c..; |wh, ww => ... | oh, ow, ..c.."
          [ "wh"; "ww" ] (stretch 0.0)
  in
  sum /. (dim wh *. dim ww)

(** Global average pooling - reduces each feature map to a single value by averaging. Commonly used
    before final classification layer. *)
let%op global_avg_pool2d x = x ++ "... | h, w, ..c.. => ... | 0, 0, ..c.."

(** Batch normalization for CNN layers - normalizes across the batch dimension for each channel.
    Typically applied after convolutions and before activations. [_momentum] is caller-visible as
    unimplemented: running statistics do not exist yet, so changing it has no effect. *)
let%op batch_norm2d ~label ?(epsilon = 1e-5) ?(_momentum = 0.9) () ~train_step x =
  (* FIXME: implement running statistics, currently using learned params *)
  (* Compute batch statistics across batch and spatial dimensions for each channel *)
  let total_size = dim o *. dim h *. dim w in
  let mean = (x ++ "..o.. | h, w, ..c.. => 0 | 0, 0, ..c.." [ "o"; "h"; "w" ]) /. total_size in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ "... | h, w, ..c.. => 0 | 0, 0, ..c..") /. total_size in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  (* Scale and shift with learnable parameters *)
  match train_step with
  | Some _ ->
      (* During training: update running statistics *)
      ({ gamma = 1. } *. normalized) + { beta = 0. }
  | None ->
      (* During inference: use running statistics (simplified for now) *)
      (gamma *. normalized) + beta

(** Batch normalization for MLP layers - normalizes across the batch axis only. Unlike
    {!batch_norm2d} there are no spatial axes to reduce over; channel axes are carried through
    unchanged via the [..c..] row variable.

    See the FIXME on {!batch_norm2d}: running statistics are not implemented, so [_momentum] is a
    caller-visible unimplemented option and inference falls back to the learned [gamma]/[beta]
    parameters rather than population statistics. Acceptable for tutorial examples; do not rely on
    inference correctness for distribution-shifted inputs. *)
let%op batch_norm1d ~label ?(epsilon = 1e-5) ?(_momentum = 0.9) () ~train_step x =
  (* Compute batch statistics across the batch axis only, for each channel *)
  let mean = (x ++ "..o.. | ..c.. => 0 | ..c.." [ "o" ]) /. dim o in
  let centered = x - mean in
  let variance = ((centered *. centered) ++ "..o.. | ..c.. => 0 | ..c..") /. dim o in
  let std_dev = sqrt (variance + !.epsilon) in
  let normalized = centered /. std_dev in
  match train_step with
  | Some _ -> ({ gamma = 1. } *. normalized) + { beta = 0. }
  | None -> (gamma *. normalized) + beta

(** Conv block with conv -> batch norm -> activation pattern *)
let%op conv_bn_relu ~label ?(kernel_size = 3) ?(stride = 1) () =
  let conv = conv2d ~label:("conv" :: label) ~kernel_size ~stride () in
  let bn = batch_norm2d ~label:("bn" :: label) () in
  fun ~train_step x -> relu (bn ~train_step (conv x))

(** Residual block for ResNet-style architectures. Features skip connections that help with gradient
    flow in deep networks. *)
let%op resnet_block ~label ?(stride = 1) () =
  let conv1 = conv2d ~label:("conv1" :: label) ~kernel_size:3 ~stride () in
  let bn1 = batch_norm2d ~label:("bn1" :: label) () in
  let conv2 = conv2d ~label:("conv2" :: label) ~kernel_size:3 ~stride:1 () in
  let bn2 = batch_norm2d ~label:("bn2" :: label) () in
  let identity =
    if Int.( > ) stride 1 then
      (* Need to downsample the skip connection *)
      let downsample_conv = conv2d ~label:("downsample" :: label) ~kernel_size:1 ~stride () in
      let downsample_bn = batch_norm2d ~label:("downsample_bn" :: label) () in
      fun train_step x -> downsample_bn ~train_step (downsample_conv x)
    else fun _train_step x -> x
  in
  fun ~train_step x ->
    let out = conv1 x |> bn1 ~train_step |> relu |> conv2 |> bn2 ~train_step in
    relu (out + identity train_step x)

(** LeNet-style architecture for simple image classification (e.g., MNIST). Classic architecture:
    conv -> pool -> conv -> pool -> fc layers. Output shape is inferred from training data. *)
let%op lenet ?(label = [ "lenet" ]) ?(out_channels1 = 6) ?(out_channels2 = 16) ?(use_padding = true)
    () =
  let conv1 =
    conv2d ~label:("conv1" :: label) ~kernel_size:5 ~use_padding ~out_channels:out_channels1 ()
  in
  let pool1 = max_pool2d ~stride:2 () in
  let conv2 =
    conv2d ~label:("conv2" :: label) ~kernel_size:5 ~use_padding ~out_channels:out_channels2 ()
  in
  let pool2 = max_pool2d ~stride:2 () in
  let fc1 = mlp_layer ~label:("fc1" :: label) ~hid_dim:120 () in
  let fc2 = mlp_layer ~label:("fc2" :: label) ~hid_dim:84 () in
  fun ~train_step:_ x ->
    let x = conv1 x |> relu |> pool1 |> conv2 |> relu |> pool2 |> fc1 |> fc2 in
    (* Final classification layer - output shape inferred from training data *)
    ({ w_logits } * x) + { b_logits = 0. }

(** VGG-style block - multiple convolutions with same filter count followed by pooling *)
let%op vgg_block ~label ~num_convs ?(kernel_size = 3) () =
  let convs =
    List.init num_convs ~f:(fun i ->
        conv_bn_relu ~label:(("conv" ^ Int.to_string i) :: label) ~kernel_size ())
  in
  let pool = max_pool2d ~stride:2 () in
  fun ~train_step x ->
    let x = List.fold convs ~init:x ~f:(fun x conv -> conv ~train_step x) in
    pool x

(** Simple CNN for Sokoban-like grid environments. Processes grid states with multiple conv layers
    and outputs action logits. *)
let%op sokoban_cnn ~label ?(num_actions = 4) () =
  (* Process spatial features with conv layers *)
  let conv1 = conv_bn_relu ~label:("conv1" :: label) ~kernel_size:3 () in
  let conv2 = conv_bn_relu ~label:("conv2" :: label) ~kernel_size:3 () in
  let conv3 = conv_bn_relu ~label:("conv3" :: label) ~kernel_size:3 () in
  fun ~train_step ~grid_state ->
    let x = conv1 ~train_step grid_state |> conv2 ~train_step |> conv3 ~train_step in

    (* Global pooling to aggregate spatial info *)
    let x = global_avg_pool2d x in

    (* Action head *)
    let action_logits = ({ w_action } * x) + { b_action = 0.; o = [ num_actions ] } in

    (* Optional: value head for actor-critic methods *)
    let value = ({ w_value } * x) + { b_value = 0.; o = [ 1 ] } in

    (action_logits, value)

(** Modern CNN with depthwise separable convolutions for efficiency. Suitable for mobile/edge
    deployment. [_width_mult] is caller-visible as unimplemented: changing it currently has no
    effect on channel counts. *)
let%op mobile_cnn ~label ?(num_classes = 1000) ?(_width_mult = 1.0) () =
  (* TODO: implement channel width multiplier *)
  (* Initial standard conv *)
  let conv_init = conv_bn_relu ~label:("conv_init" :: label) ~kernel_size:3 ~stride:2 () in

  (* Depthwise separable blocks *)
  let dw_block1 = depthwise_separable_conv2d ~label:("dw1" :: label) ~stride:1 () in
  let dw_block2 = depthwise_separable_conv2d ~label:("dw2" :: label) ~stride:2 () in
  let dw_block3 = depthwise_separable_conv2d ~label:("dw3" :: label) ~stride:1 () in
  let dw_block4 = depthwise_separable_conv2d ~label:("dw4" :: label) ~stride:2 () in

  let bn = batch_norm2d ~label:("bn_final" :: label) () in

  fun ~train_step x ->
    let x =
      conv_init ~train_step x |> dw_block1 |> relu |> dw_block2 |> relu |> dw_block3 |> relu
      |> dw_block4 |> relu |> bn ~train_step
    in

    (* Global pooling and classification *)
    let x = global_avg_pool2d x in
    ({ w_classifier } * x) + { b_classifier = 0.; o = [ num_classes ] }
