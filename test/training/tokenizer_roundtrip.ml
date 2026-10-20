(* Integration test for the dataprep BPE tokenizer -> OCANNL tensor bridge
   (docs/proposals/dataprep-tokenizer-integration.md).

   Hermetic: instead of [Dataprep.Bpe.from_pretrained] (which downloads from HuggingFace), the test
   writes a minimal GPT-2-style byte-level BPE tokenizer.json and loads it with [Dataprep.Bpe.load].
   It then encodes text, bridges the token IDs to OCANNL tensors via [Nn_blocks.token_ids_of_array]
   / [token_ids_of_batch] (with padding and truncation), reads the values back through a device
   roundtrip, and checks composition with the one-hot embedding-lookup path. *)

open Base
open Ocannl
open Stdio
open Nn_blocks.DSL_modules
module Tn = Ir.Tnode
module Bpe = Dataprep.Bpe

(* "Ġ" (U+0120) is the GPT-2 byte-level encoding of a space. *)
let tokenizer_json =
  {|{
  "model": {
    "type": "BPE",
    "vocab": { "a": 0, "b": 1, "ab": 2, "Ġab": 3, "Ġ": 4 },
    "merges": [ "a b", "Ġ ab" ]
  }
}|}

(* The ID of the lone-space token "Ġ": used as the padding token below. *)
let pad_id = 4
let ints_str ids = String.concat ~sep:"; " (Array.to_list (Array.map ids ~f:Int.to_string))
let vals_str vals = ints_str (Array.map vals ~f:Float.to_int)

let dims_str t =
  String.concat ~sep:"x"
    (Array.to_list (Array.map (Lazy.force t.Tensor.value.Tn.dims) ~f:Int.to_string))

let prec_str t = Ir.Ops.prec_string (Lazy.force t.Tensor.value.Tn.storage_prec)

let () =
  Out_channel.write_all "mini_tokenizer.json" ~data:tokenizer_json;
  let tok = Bpe.load "mini_tokenizer.json" in
  printf "vocab size: %d\n" (Bpe.vocab_size tok);
  let text = "ab ab ab" in
  let ids = Bpe.encode tok text in
  printf "encode %S -> [%s]\n" text (ints_str ids);
  printf "decode roundtrip: %S\n" (Bpe.decode tok ids);

  (* As-is conversion: a [len] batch of uint32 token IDs. *)
  let ctx = Context.auto () in
  let t = Nn_blocks.token_ids_of_array ids in
  let ctx = Train.forward_once ctx t in
  let vals = Context.get_values ctx t.Tensor.value in
  printf "tensor: dims %s, prec %s, values [%s]\n" (dims_str t) (prec_str t) (vals_str vals);

  (* Padding to a fixed sequence length; decode what the tensor actually holds. *)
  let padded = Nn_blocks.token_ids_of_array ~max_len:5 ~pad_id ids in
  let ctx = Train.forward_once ctx padded in
  let padded_vals = Context.get_values ctx padded.Tensor.value in
  printf "padded to 5: dims %s, values [%s]\n" (dims_str padded) (vals_str padded_vals);
  printf "decode of padded tensor contents: %S\n"
    (Bpe.decode tok (Array.map padded_vals ~f:Float.to_int));

  (* A single-token sequence must keep its [1] batch axis (a Reshape-inferred row would collapse
     total-elements-1 to a scalar), so that e.g. one-hot composition yields [1; vocab]. *)
  let single = Nn_blocks.token_ids_of_array (Bpe.encode tok "ab") in
  let single_oh = Nn_blocks.one_hot_of_ids ~num_classes:(Bpe.vocab_size tok) single in
  let ctx = Train.forward_once ctx single_oh in
  let single_oh_vals = Context.get_values ctx single_oh.Tensor.value in
  printf "single token: dims %s, one-hot dims %s, values [%s]\n" (dims_str single)
    (dims_str single_oh) (vals_str single_oh_vals);

  (* Truncation. *)
  let truncated = Nn_blocks.token_ids_of_array ~max_len:2 ids in
  let ctx = Train.forward_once ctx truncated in
  let truncated_vals = Context.get_values ctx truncated.Tensor.value in
  printf "truncated to 2: dims %s, values [%s]\n" (dims_str truncated) (vals_str truncated_vals);

  (* Batched: sequences of different lengths, padded to the longest by default. *)
  let seqs = [| Bpe.encode tok "ab"; Bpe.encode tok text |] in
  let batch = Nn_blocks.token_ids_of_batch ~pad_id seqs in
  let ctx = Train.forward_once ctx batch in
  let batch_vals = Context.get_values ctx batch.Tensor.value in
  printf "batch: dims %s, prec %s, values [%s]\n" (dims_str batch) (prec_str batch)
    (vals_str batch_vals);

  (* The batched IDs compose with the embedding-lookup path: a logical one-hot over the vocab,
     shaped [num_seqs; max_len; vocab_size]. *)
  let vocab = Bpe.vocab_size tok in
  let oh = Nn_blocks.one_hot_of_ids ~num_classes:vocab batch in
  let ctx = Train.forward_once ctx oh in
  let oh_vals = Context.get_values ctx oh.Tensor.value in
  let oh_expected =
    Array.concat_map batch_vals ~f:(fun id ->
        Array.init vocab ~f:(fun k -> if k = Float.to_int id then 1. else 0.))
  in
  Verdict.pf_close "one-hot of batch: dims %s, matches the token ids" (dims_str oh) oh_vals
    oh_expected ~tol:(fun _ -> 1e-6)
