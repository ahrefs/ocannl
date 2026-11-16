(* gh-ocannl-2192: a source mark must not strand another device behind an unguarded wait. *)
let () = Wait_wedge_fixture.cross_device ()
