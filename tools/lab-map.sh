#!/usr/bin/env bash
# Lab endpoint lookups and lock-contract checks, sourced by tools/sweep.sh.
# The caller owns LAB_MAP, WAKE_LAB, LAB_HOSTS, LAB_LOCK_DIR and LAB_LANE_BOXES,
# and provides ask_capped and die. Sourcing defines functions only.

# ------------------------------------------------------------------- the lab's endpoint map
# Which box an ssh alias belongs to, and which alias reaches a box's native Ubuntu or its WSL guest,
# are wake-lab.sh's to say: its ENDPOINT_MAP is the one box -> endpoint table the lab has
# (ludics-lite#314), and `wake-lab.sh endpoint-map` prints it as data, a box name and then that box's
# ssh aliases on each line (ludics-lite#395), answering only for a map its own row rules pass. The
# sweep reads it once, at startup, whenever a remote unit is selected, and keeps no table of its own
# (gh-ocannl-1121). Two hand-written tables beside it, a startup check that the three agree, and a
# fallback for when the map could not be read were each a restatement of that one table, and the
# fallback alone took three review rounds to keep complete (staging#868). No map means no remote
# unit, and the run refuses at startup: a host without wake-lab.sh is one without the site host
# table too, which lab_dest already refuses a remote unit without, and one without a destroyer
# to coordinate with.
#
# The map carries no OS keys, only aliases; the OS is the alias's suffix. That is wake-lab's own row
# rule (check_endpoints: `<stem>-linux`, `<stem>-win`, `<stem>-wsl`, and `<box>-lan`), and the
# suffix is what the rest of this script already reads the boot from -- the remote PATH below
# run_unit's preparation, and box_jobs_dest_transport's `dxg`/`native` -- so nothing here restates
# an alias.
lab_map() { # -- sets LAB_MAP from wake-lab.sh, or refuses the run
  [ -x "$WAKE_LAB" ] ||
    die "no wake-lab.sh at $WAKE_LAB, whose endpoint map names every remote unit's ssh aliases (set OCANNL_TOOL_SWEEP_WAKE_LAB)"
  ask_capped LAB_MAP 60 "$WAKE_LAB" endpoint-map || LAB_MAP=
  [ -n "$LAB_MAP" ] ||
    die "$WAKE_LAB endpoint-map gave no map (a wake-lab.sh from before ludics-lite#395?), so no remote unit has an ssh alias"
}

lab_row() { # box -- that box's aliases, space-separated; 1 when the map has no row for it
  awk -v b="$1" '$1 == b { $1 = ""; sub(/^ +/, ""); print; found = 1; exit } END { exit !found }' \
    <<<"$LAB_MAP"
}

# The wake-lab box whose lock covers an ssh alias: the box whose row lists it. Both boots of a box
# share ONE lock -- the box, not the OS it booted, is what a restart or a power verb takes away
# (gh-ocannl-1030) -- and wake-lab's map refuses an alias listed on two rows (check_map).
lab_box_of() { # ssh-alias -- 1 when no row lists it
  awk -v a="$1" '{ for (i = 2; i <= NF; i++) if ($i == a) { print $1; found = 1; exit } }
    END { exit !found }' <<<"$LAB_MAP"
}

lab_dest_of() { # box kind -- the ssh alias for that boot of that box; 1 when its row has none
  local alias
  case $2 in linux | wsl) ;; *) return 1 ;; esac
  for alias in $(lab_row "$1"); do
    case $alias in *-"$2") printf '%s' "$alias"; return 0 ;; esac
  done
  return 1
}

# Prints the box's destination, or says on stderr why there is none and returns 1. The table is
# sourced in a SUBSHELL: it is site shell code, and nothing it defines may reach this script's own
# functions. Its stdout is discarded so that only kind_of's answer is read as the kind.
lab_dest() { # box
  local box=$1 var dest kind row
  var=OCANNL_TOOL_SWEEP_DEST_$(printf '%s' "$box" | tr '[:lower:]' '[:upper:]')
  if ! row=$(lab_row "$box"); then
    echo "sweep: $WAKE_LAB endpoint-map has no row for $box" >&2
    return 1
  fi
  dest=${!var:-}
  if [ -n "$dest" ]; then
    # Membership, not a shape check: that closes every way an override could mean something the
    # rest of the script reads differently -- an alias of another box (the lane would reserve the
    # wrong lock), the box's Windows or LAN route (no Linux to run a suite on), a `user@` or a second
    # `@` (the host ssh contacts and the lock lab_box_of derives become two parses of one string),
    # an option-shaped word. A user or a different address belongs in the alias's ssh config.
    for kind in linux wsl; do
      if [ "$dest" = "$(lab_dest_of "$box" "$kind")" ]; then
        printf '%s' "$dest"
        return 0
      fi
    done
    echo "sweep: $var='$dest' is not the -linux or -wsl alias on $box's row ($row)" >&2
    return 1
  fi
  if [ ! -r "$LAB_HOSTS" ]; then
    echo "sweep: cannot read the site host table $LAB_HOSTS for $box's boot kind" \
      "(set WAKE_LAB_HOSTS, or name the destination with $var)" >&2
    return 1
  fi
  # The kind_of consulted must be the TABLE's: bash imports an exported function from the
  # environment before this script starts, and an inherited kind_of would otherwise answer for a
  # table that defines none -- the refusal below exists for exactly that table.
  kind=$(
    set +u
    unset -f kind_of
    # shellcheck source=/dev/null
    . "$LAB_HOSTS" >/dev/null </dev/null || exit 1
    declare -F kind_of >/dev/null || exit 1
    kind_of "$box" </dev/null
  ) || kind=
  case $kind in
    linux | wsl) ;;
    *)
      echo "sweep: the site host table $LAB_HOSTS gives no usable boot kind for $box" \
        "(kind_of $box: '${kind:-<none>}'; expected linux or wsl)" >&2
      return 1
      ;;
  esac
  if ! dest=$(lab_dest_of "$box" "$kind"); then
    echo "sweep: $WAKE_LAB endpoint-map lists no -$kind alias for $box, whose boot kind is $kind" \
      "(its row: $row)" >&2
    return 1
  fi
  printf '%s' "$dest"
}

lab_contract_check() { # -- sets LAB_CONTRACT; refuses the run on a broken contract
  local box path want broken=
  [ -n "$LAB_LANE_BOXES" ] || return 0
  for box in $LAB_LANE_BOXES; do
    want=$LAB_LOCK_DIR/$box.lock
    ask_capped path 60 "$WAKE_LAB" lock-path "$box" || path=
    [ "$path" = "$want" ] ||
      broken="$broken; lock-path $box answers '$path' where a lane locks $want"
  done
  [ -z "$broken" ] ||
    die "the lab lock contract with $WAKE_LAB is broken${broken/#;/:}; a lane would reserve a box no destroyer checks, so fix the side that moved"
  LAB_CONTRACT="agree with $WAKE_LAB for $LAB_LANE_BOXES"
}

