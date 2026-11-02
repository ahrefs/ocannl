(** The ppx library's entry module: registers the [%cd], [%op] and [%%extend_dsls] expanders with
    ppxlib as the [ppx_ocannl] transformation. *)

val schemed_definitions : Ppxlib.structure -> (string * bool) list
(** {!Ppx_op.schemed_definitions}, exposed for tests pinning which definitions get a scheme. *)
