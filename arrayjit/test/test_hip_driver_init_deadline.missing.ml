(* Selected where hipjit is unavailable. The test's subject is then absent for an environment
   reason, not a verdict about whichever runtime backend was requested. *)

let claim =
  "a stuck HIP driver initialization that a direct Impl.hip_to_code call forces dies by the \
   device-init deadline, with the diagnostic naming hip"

let () = Verdict.skipped ~aggregation:`Environment ~backend:"hipjit unavailable" claim
