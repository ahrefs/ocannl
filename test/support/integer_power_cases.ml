(* The integer-power domain (gh-ocannl-1185): the bases and known integral exponents
   [integer_power_domain] executes through the backend, and over which
   [runtime_power_fast_math_emulation] compares the CUDA-like [ocannl_powi_*] helpers bitwise to the
   plain C ones. Dyadic bases keep every finite answer exact; the exceptional values are the point.
   Stdlib only, so the GPU-free emulation test links nothing more. *)

let bases = [ -2.; -0.5; -1.; 0.; -0.; Float.infinity; Float.neg_infinity; Float.nan ]
let exponents = [ 0; 1; 2; 3; -1; -2; -3; 8; -8; 31; -31 ]

(* Exponents outside every machine int range, as [(base, exponent, power)]: they still retain parity
   and sign, and all powers of +/-1 are exact even at these magnitudes. *)
let large_exponents =
  [
    (-1., 0x1p63, 1.);
    (-1., -0x1p63, 1.);
    (-1., 0x1.fffffffffffffp52, -1.);
    (-1., 0x1.fffffffffffffp1023, 1.);
    (-1., -0x1.fffffffffffffp1023, 1.);
  ]
