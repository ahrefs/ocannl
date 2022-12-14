(** Losses and the training loop. *)

let hinge_loss m y = Model.O.(m @> fun score -> !/(!.1.0 - y * score))
