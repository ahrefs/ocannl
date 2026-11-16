(** The compilation units a compiled module's typed tree names (gh-ocannl-2118); see the
    implementation's header for what counts as a reference. *)

val referenced_units : string -> (string list, string) result
(** The persistent units at the head of every path the typed tree of one [.cmt] or [.cmti] file
    resolves, sorted. [Error] when the file cannot be read or holds no complete typed tree. *)
