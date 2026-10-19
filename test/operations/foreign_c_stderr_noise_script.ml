(* gh-ocannl-1370: the writes [foreign_c_stderr_noise_child] makes on the C runtime's [stderr], one
   [fputs] each, and what the collapsing stream must make of them. Shared by the child, which writes
   them, and the parent, which reads back what reached the child's stderr. *)

let begin_marker = "foreign-begin\n"
let end_marker = "foreign-end\n"
let recognized addr = "Signal 0x" ^ addr ^ " time stamps may be invalid.\n"

(* Six recognized lines (100, 200, 300 in upper case, 500 split over three writes, 600, and 700 in
   the middle of a write), and around them every near miss the matcher must write through: a
   different ending, a non-hex address, trailing text, a leading prefix, and a held prefix that
   turns out not to be the line. *)
let writes =
  [
    begin_marker;
    "an ordinary foreign line\n";
    recognized "7f3a1c000100";
    recognized "7f3a1c000200";
    "between two recognized lines\n";
    recognized "7F3A1C000300";
    "Signal 0x7f3a1c000400 time stamps may be valid.\n";
    "Signal 0xZZ time stamps may be invalid.\n";
    "Signal 0x1 time stamps may be invalid. (and more)\n";
    "a prefix, then Signal 0x1 time stamps may be invalid.\n";
    "Signal 0x7f3a";
    "1c000500 time stamps";
    " may be invalid.\n";
    "Signal 0x12";
    "34 is held, then written whole\n";
    recognized "7f3a1c000600";
    "two lines in one write\n" ^ recognized "7f3a1c000700" ^ "and the line after\n";
    end_marker;
  ]

let recognized_count = 6

(* What the collapsing stream writes, without its own two [OCANNL:] lines: every write verbatim and
   in order, except that only the first recognized line survives. *)
let collapsed_lines =
  [
    "foreign-begin";
    "an ordinary foreign line";
    "Signal 0x7f3a1c000100 time stamps may be invalid.";
    "between two recognized lines";
    "Signal 0x7f3a1c000400 time stamps may be valid.";
    "Signal 0xZZ time stamps may be invalid.";
    "Signal 0x1 time stamps may be invalid. (and more)";
    "a prefix, then Signal 0x1 time stamps may be invalid.";
    "Signal 0x1234 is held, then written whole";
    "two lines in one write";
    "and the line after";
    "foreign-end";
  ]
