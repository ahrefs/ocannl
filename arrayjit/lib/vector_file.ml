open Base

type t = { bytes : int; registers : int } [@@deriving sexp, compare, equal]

let none = { bytes = 0; registers = 0 }
