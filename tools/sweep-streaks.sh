#!/usr/bin/env bash
# Detect persistent skip-coverage and bypass-rate streaks across consecutive
# saved sweep reports (gh-ocannl-2424).
#
# A single-night skip or contention bypass can come from a box outage or a busy
# host, while the same claim skipped on every box, the same claim UNVERIFIED, or
# the same unit at 100% bypass across N consecutive sweeps (default N = 3) is
# coverage that silently stopped. This script compares saved reports in
# chronological order and flags any trailing streak of length >= N.
#
# What it reads, as a fail-closed allowlist:
# - Saved skip-coverage reports (`*-skip-coverage.txt`), produced by
#   `tools/aggregate-skips.sh` (optionally prefixed with `tools/sweep.sh`'s
#   `skip coverage for <sha> (<scope>; forced execution)` and
#   `red units counted after a clean serial rerun: ...` lines). Every line must
#   match the aggregator's output grammar; required status/result lines must be
#   present and mutually consistent; declared FAIL/POTENTIAL counts must match
#   the listed finding lines. Legacy pre-environment reports (predating the
#   `completed boxes:` / `environment status:` / `environment result:` block)
#   are accepted with the environment dimension unjudged.
# - Saved bypass-rate reports (`*-bypass-rates.txt`), produced by
#   `tools/aggregate-bypasses.sh` (optionally prefixed with `tools/sweep.sh`'s
#   `bypass rates for <sha> (<forced|incremental> execution)` line). Every line
#   must match the aggregator's output grammar; every declared unit must appear
#   once; bypassed + checked must equal reaches with consistent rounded percent
#   and indented per-claim bypass totals; declared UNVERIFIED counts must match
#   the listed UNVERIFIED lines.
# - When reading directories (`--logs-dir`, `--skip-logs-dir`,
#   `--bypass-logs-dir`), only regular files matching
#   `YYYYMMDDTHHMMSSZ-skip-coverage.txt` and `YYYYMMDDTHHMMSSZ-bypass-rates.txt`
#   (up to `--up-to STAMP` if given) are read, in bytewise UTC stamp order;
#   other files (staging files, logs, run records) are ignored.
# Any unrecognized or truncated line, missing required section, duplicate unit
# or claim within one report, count mismatch, unreadable file or directory, or
# invalid option refuses the run: exit 2 normally (and always for the report
# matching `--up-to STAMP`), or exit 3 when `--up-to STAMP` is given and an
# OLDER historical report (< STAMP) is malformed so `tools/sweep.sh` can warn
# loudly without failing every subsequent nightly sweep as `post-run-failed`.
#
# Streak rules (default min-streak N = 3; minimum permitted N = 2):
# Note on cadence: N counts consecutive judged REPORTS in each family, not
# calendar nights. Bypass-rate reports are written by every unscoped sweep
# (daily, so N = 3 is ~3 nights); skip-coverage reports are written only by
# forced unscoped sweeps (weekly on Sundays, so N = 3 is ~3 forced sweeps / ~3
# weeks at the current cadence). Same-day reruns each write a report and count
# as one sweep in that family's sequence (not collapsed by UTC date).
# 1. `skipped on every box`: `FAIL: skipped on every declared box: ...` and
#    `FAIL: skipped on every required sweep box: ...` normalize to one box-skip
#    dimension so streaks spanning `--unswept-box` adoption do not split.
#    - `environment status: complete`: listed FAIL claims advance (+1); all
#      other claims clear (0).
#    - `environment status: partial`: claims listed in `POTENTIAL` are
#      inconclusive (neither advance nor clear); claims absent from `POTENTIAL`
#      were evaluated on some completed box, so they clear (0).
#    - `environment status: insufficient`/`unavailable` or legacy report:
#      inconclusive for all claims.
# 2. `skipped on every known backend`: `FAIL: skipped on every known backend: ...`
#    follows the symmetric rules under `status:` and backend `POTENTIAL`.
# 3. `unit at 100% bypass`: `<unit>: <B> of <R> ... (<P>%), <C> checked` with
#    `R > 0`, `B == R`, and `C == 0` (judged on exact counts, never the rounded
#    percentage, so `199 of 200 (100%), 1 checked` clears).
#    - `B == R && C == 0`: advances (+1).
#    - `C > 0`: clears (0).
#    - `<unit>: no hatched claim reached`, unit absent, or `NOT AGGREGATED`:
#      inconclusive for that unit.
# 4. `UNVERIFIED bypassed claim`: `UNVERIFIED: bypassed on every unit that
#    reached it: <exe>: <claim> (<units>)`, keyed by `<exe>: <claim>`.
#    - Listed under `UNVERIFIED:`: advances (+1).
#    - `result: CLEAR -- no hatched claim was reached` or `NOT AGGREGATED`:
#      inconclusive for all claims.
#    - `result: CLEAR -- every hatched claim reached was checked on some unit`,
#      or `result: UNVERIFIED` not naming the claim: clears (0). Because
#      `tools/aggregate-bypasses.sh` omits `  bypassed` detail lines when a
#      claim has 0 bypasses (`b == 0`), a claim that ran and was 100% checked
#      appears nowhere in the detail lines either; treating absence from detail
#      lines as inconclusive would fail to clear streaks across 0%-bypass nights.
# Activation: a streak is flagged only when the LATEST report in that family is
# itself a hit for the item and its trailing judged streak is >= N. Intervening
# inconclusive reports neither advance nor reset a streak, and a run that did
# not observe the item (e.g. `--only cc` or a partial sweep) does not alert.
#
# What it deliberately does not do: read raw `*.verdict-records`, sweep logs,
# `history.tsv`, or `unit-state/`; open or update GitHub issues (the scheduled
# sweep routine files/updates one standing issue per streak for the CI-red
# triage routine to read); or fail `tools/sweep.sh` on a finding.
#
# Usage:
#   tools/sweep-streaks.sh [--min-streak N] --logs-dir DIR [--up-to STAMP]
#   tools/sweep-streaks.sh [--min-streak N] \
#     [--skip-logs-dir DIR] [--bypass-logs-dir DIR] [--up-to STAMP]
#   tools/sweep-streaks.sh [--min-streak N] \
#     [--skip-report FILE]... [--bypass-report FILE]...
#
# Exit 0 when no streak reached min-streak (or no reports were supplied);
# 1 when one or more streaks reached min-streak; 2 on CLI/I/O error or a
# malformed report (including the `--up-to STAMP` report itself); 3 when
# `--up-to STAMP` is given and only an older historical report (< STAMP) is
# malformed.

set -uo pipefail
# Name a locale before any subshell forks (gh-ocannl-2002, gh-ocannl-2462, gh-ocannl-2484).
[ -n "${LANG:-}" ] || [ -n "${LC_ALL:-}" ] || export LANG=C

exec perl -e '
use strict;
use warnings;

our $MALFORMED_EXIT = 2;
our $MALFORMED_KIND = "";

sub die_input {
  my ($msg) = @_;
  print STDERR "sweep-streaks: $msg\n";
  exit 2;
}

sub die_malformed {
  my ($msg) = @_;
  if ($MALFORMED_KIND ne "") {
    $msg =~ s/^malformed /malformed $MALFORMED_KIND /;
  }
  print STDERR "sweep-streaks: $msg\n";
  exit $MALFORMED_EXIT;
}

sub write_line {
  my ($line) = @_;
  print STDOUT "$line\n" or die_input("cannot write report");
}

my $min_streak = 3;
my $logs_dir;
my $skip_logs_dir;
my $bypass_logs_dir;
my $up_to;
my @skip_reports;
my @bypass_reports;
my %seen_skip_arg;
my %seen_bypass_arg;

my $i = 0;
while ($i < @ARGV) {
  my $arg = $ARGV[$i++];
  if ($arg eq "--min-streak") {
    $i < @ARGV or die_input("--min-streak needs an integer >= 2");
    my $val = $ARGV[$i++];
    $val =~ /^[0-9]{1,9}$/ && $val >= 2
      or die_input("--min-streak must be an integer >= 2, not '\''$val'\''");
    $min_streak = int($val);
  } elsif ($arg eq "--logs-dir") {
    $i < @ARGV or die_input("--logs-dir needs a directory");
    defined $logs_dir and die_input("--logs-dir given more than once");
    $logs_dir = $ARGV[$i++];
    $logs_dir ne "" or die_input("--logs-dir needs a non-empty path");
  } elsif ($arg eq "--skip-logs-dir") {
    $i < @ARGV or die_input("--skip-logs-dir needs a directory");
    defined $skip_logs_dir and die_input("--skip-logs-dir given more than once");
    $skip_logs_dir = $ARGV[$i++];
    $skip_logs_dir ne "" or die_input("--skip-logs-dir needs a non-empty path");
  } elsif ($arg eq "--bypass-logs-dir") {
    $i < @ARGV or die_input("--bypass-logs-dir needs a directory");
    defined $bypass_logs_dir and die_input("--bypass-logs-dir given more than once");
    $bypass_logs_dir = $ARGV[$i++];
    $bypass_logs_dir ne "" or die_input("--bypass-logs-dir needs a non-empty path");
  } elsif ($arg eq "--up-to") {
    $i < @ARGV or die_input("--up-to needs a UTC stamp");
    defined $up_to and die_input("--up-to given more than once");
    $up_to = $ARGV[$i++];
    $up_to =~ /^[0-9]{8}T[0-9]{6}Z$/
      or die_input("--up-to must be a UTC stamp YYYYMMDDTHHMMSSZ, not '\''$up_to'\''");
  } elsif ($arg eq "--skip-report") {
    $i < @ARGV or die_input("--skip-report needs a file path");
    my $path = $ARGV[$i++];
    $path ne "" or die_input("--skip-report needs a non-empty path");
    die_input("duplicate --skip-report '\''$path'\''") if $seen_skip_arg{$path}++;
    push @skip_reports, $path;
  } elsif ($arg eq "--bypass-report") {
    $i < @ARGV or die_input("--bypass-report needs a file path");
    my $path = $ARGV[$i++];
    $path ne "" or die_input("--bypass-report needs a non-empty path");
    die_input("duplicate --bypass-report '\''$path'\''") if $seen_bypass_arg{$path}++;
    push @bypass_reports, $path;
  } else {
    die_input("unknown argument: $arg");
  }
}

if (defined $logs_dir) {
  die_input("cannot combine --logs-dir with --skip-logs-dir or --bypass-logs-dir")
    if defined $skip_logs_dir || defined $bypass_logs_dir;
  $skip_logs_dir = $logs_dir;
  $bypass_logs_dir = $logs_dir;
}

die_input("cannot combine --skip-logs-dir/--logs-dir with --skip-report")
  if defined $skip_logs_dir && @skip_reports;
die_input("cannot combine --bypass-logs-dir/--logs-dir with --bypass-report")
  if defined $bypass_logs_dir && @bypass_reports;
die_input("--up-to requires --logs-dir, --skip-logs-dir, or --bypass-logs-dir")
  if defined $up_to && !defined $skip_logs_dir && !defined $bypass_logs_dir;

sub discover_reports {
  my ($dir, $suffix, $limit) = @_;
  opendir(my $dh, $dir) or die_input("cannot open logs directory $dir: $!");
  my @matches;
  while (defined(my $entry = readdir($dh))) {
    next unless $entry =~ /^([0-9]{8}T[0-9]{6}Z)\Q$suffix\E$/;
    my $stamp = $1;
    next if defined $limit && $stamp gt $limit;
    my $full = "$dir/$entry";
    next unless -f $full;
    push @matches, [$stamp, $full];
  }
  closedir($dh);
  return map { { path => $_->[1], stamp => $_->[0] } } sort { $a->[0] cmp $b->[0] } @matches;
}

my @skip_items;
my @bypass_items;
if (defined $skip_logs_dir) {
  @skip_items = discover_reports($skip_logs_dir, "-skip-coverage.txt", $up_to);
} else {
  @skip_items = map { { path => $_, stamp => undef } } @skip_reports;
}
if (defined $bypass_logs_dir) {
  @bypass_items = discover_reports($bypass_logs_dir, "-bypass-rates.txt", $up_to);
} else {
  @bypass_items = map { { path => $_, stamp => undef } } @bypass_reports;
}
@skip_reports   = map { $_->{path} } @skip_items;
@bypass_reports = map { $_->{path} } @bypass_items;

sub read_lines {
  my ($path) = @_;
  open(my $fh, "<", $path) or die_input("cannot read $path: $!");
  my @lines;
  while (my $line = <$fh>) {
    chomp $line;
    push @lines, $line;
  }
  close($fh) or die_input("cannot close $path: $!");
  return \@lines;
}

sub parse_skip_report {
  my ($path) = @_;
  my $lines = read_lines($path);
  @$lines > 0 or die_malformed("malformed skip report $path: empty file");

  my $phase = "header";
  my ($seen_header_for, $seen_rerun) = (0, 0);
  my ($seen_comp_b, $seen_miss_b, $b_status, $b_result, $b_expected) = (0, 0, undef, undef, 0);
  my ($seen_comp_e, $seen_miss_e, $seen_unswept, $seen_undeclared) = (0, 0, 0, 0);
  my ($e_status, $e_result, $e_expected, $e_box_every) = (undef, undef, 0, undef);
  my (%b_fail, %b_pot, %e_fail, %e_pot);
  my ($b_count, $e_count) = (0, 0);

  my $lno = 0;
  for my $line (@$lines) {
    $lno++;
    if ($phase eq "header") {
      if ($line =~ /^skip coverage for [0-9a-f]+ \(.+; forced execution\)$/) {
        die_malformed("malformed skip report $path:$lno: duplicate header") if $seen_header_for || $seen_rerun;
        $seen_header_for = 1;
        next;
      }
      if ($line =~ /^red units counted after a clean serial rerun: .+$/) {
        die_malformed("malformed skip report $path:$lno: duplicate rerun header") if $seen_rerun;
        $seen_rerun = 1;
        next;
      }
      if ($line =~ /^completed backends: .+$/) {
        $seen_comp_b = 1;
        $phase = "backend_meta";
        next;
      }
      die_malformed("malformed skip report $path:$lno: unexpected line before completed backends");
    }

    if ($phase eq "backend_meta") {
      if (!$seen_miss_b && $line =~ /^missing backends: .+$/) {
        $seen_miss_b = 1;
        next;
      }
      if ($seen_miss_b && !defined $b_status && $line =~ /^status: (complete|partial|insufficient) \(.+\)$/) {
        $b_status = $1;
        next;
      }
      if (defined $b_status && !defined $b_result) {
        if ($b_status eq "insufficient" && $line eq "result: NOT AGGREGATED") {
          $b_result = "NOT_AGGREGATED";
          $phase = "after_backend";
          next;
        }
        if ($b_status eq "complete" && $line =~ /^result: PASS -- .+$/) {
          $b_result = "PASS";
          $phase = "after_backend";
          next;
        }
        if ($b_status eq "complete" && $line =~ /^result: FAIL -- ([0-9]+) claim\(s\) skipped on every known backend$/ && $1 >= 1) {
          $b_result = "FAIL";
          $b_expected = int($1);
          $phase = "backend_findings";
          next;
        }
        if ($b_status eq "partial" && $line =~ /^result: CLEAR across completed backends -- .+$/) {
          $b_result = "CLEAR";
          $phase = "after_backend";
          next;
        }
        if ($b_status eq "partial" && $line =~ /^result: POTENTIAL -- ([0-9]+) claim\(s\) skipped on every completed backend; .+$/) {
          die_malformed("malformed skip report $path:$lno: POTENTIAL count must be >= 1") unless $1 >= 1;
          $b_result = "POTENTIAL";
          $b_expected = int($1);
          $phase = "backend_findings";
          next;
        }
      }
      die_malformed("malformed skip report $path:$lno: unexpected backend metadata line");
    }

    if ($phase eq "backend_findings") {
      if ($b_result eq "FAIL" && $line =~ /^FAIL: skipped on every known backend: (.+)$/) {
        my $claim = $1;
        die_malformed("malformed skip report $path:$lno: duplicate backend FAIL claim") if $b_fail{$claim}++;
        $b_count++;
        next;
      }
      if ($b_result eq "POTENTIAL" && $line =~ /^POTENTIAL: skipped on every completed backend: (.+)$/) {
        my $claim = $1;
        die_malformed("malformed skip report $path:$lno: duplicate backend POTENTIAL claim") if $b_pot{$claim}++;
        $b_count++;
        next;
      }
      die_malformed("malformed skip report $path:$lno: backend finding count mismatch (expected $b_expected, got $b_count)")
        unless $b_count == $b_expected;
      $phase = "after_backend";
      # fall through to handle this line in after_backend
    }

    if ($phase eq "after_backend") {
      if ($line =~ /^completed boxes: .+$/) {
        $seen_comp_e = 1;
        $phase = "env_meta";
        next;
      }
      if ($line =~ /^outside the sweep: .+ -- (?:owned by .+|owner not recorded \(a swept commit predating gh-ocannl-926\))$/) {
        $phase = "outside";
        next;
      }
      die_malformed("malformed skip report $path:$lno: unexpected line after backend section");
    }

    if ($phase eq "env_meta") {
      if (!$seen_miss_e && $line =~ /^missing boxes: .+$/) {
        $seen_miss_e = 1;
        next;
      }
      if ($seen_miss_e && !defined $e_status && !$seen_unswept && !$seen_undeclared &&
          $line =~ /^unswept declared boxes \(excluded from required matrix\): .+$/) {
        $seen_unswept = 1;
        next;
      }
      if ($seen_miss_e && !defined $e_status && !$seen_undeclared &&
          $line =~ /^undeclared boxes \(their executions count, their absence does not\): .+$/) {
        $seen_undeclared = 1;
        next;
      }
      if ($seen_miss_e && !defined $e_status &&
          $line =~ /^environment status: (complete|partial|insufficient|unavailable) \(.+\)$/) {
        $e_status = $1;
        next;
      }
      if (defined $e_status && !defined $e_result) {
        if (($e_status eq "insufficient" || $e_status eq "unavailable") &&
            $line eq "environment result: NOT AGGREGATED") {
          $e_result = "NOT_AGGREGATED";
          $phase = "outside";
          next;
        }
        if ($e_status eq "complete" &&
            $line =~ /^environment result: PASS -- no claim was skipped on (every declared box|every required sweep box)$/) {
          $e_result = "PASS";
          $e_box_every = $1;
          $phase = "outside";
          next;
        }
        if ($e_status eq "complete" &&
            $line =~ /^environment result: FAIL -- ([0-9]+) claim\(s\) skipped on (every declared box|every required sweep box)$/) {
          die_malformed("malformed skip report $path:$lno: environment FAIL count must be >= 1") unless $1 >= 1;
          $e_result = "FAIL";
          $e_expected = int($1);
          $e_box_every = $2;
          $phase = "env_findings";
          next;
        }
        if ($e_status eq "partial" && $line =~ /^environment result: CLEAR across completed boxes -- .+$/) {
          $e_result = "CLEAR";
          $phase = "outside";
          next;
        }
        if ($e_status eq "partial" &&
            $line =~ /^environment result: POTENTIAL -- ([0-9]+) claim\(s\) skipped on every completed box; .+$/) {
          die_malformed("malformed skip report $path:$lno: environment POTENTIAL count must be >= 1") unless $1 >= 1;
          $e_result = "POTENTIAL";
          $e_expected = int($1);
          $phase = "env_findings";
          next;
        }
      }
      die_malformed("malformed skip report $path:$lno: unexpected environment metadata line");
    }

    if ($phase eq "env_findings") {
      if ($e_result eq "FAIL" && $line =~ /^FAIL: skipped on \Q$e_box_every\E: (.+)$/) {
        my $claim = $1;
        die_malformed("malformed skip report $path:$lno: duplicate environment FAIL claim") if $e_fail{$claim}++;
        $e_count++;
        next;
      }
      if ($e_result eq "POTENTIAL" && $line =~ /^POTENTIAL: skipped on every completed box: (.+)$/) {
        my $claim = $1;
        die_malformed("malformed skip report $path:$lno: duplicate environment POTENTIAL claim") if $e_pot{$claim}++;
        $e_count++;
        next;
      }
      die_malformed("malformed skip report $path:$lno: environment finding count mismatch (expected $e_expected, got $e_count)")
        unless $e_count == $e_expected;
      $phase = "outside";
      # fall through to handle this line in outside
    }

    if ($phase eq "outside") {
      if ($line =~ /^outside the sweep: .+ -- (?:owned by .+|owner not recorded \(a swept commit predating gh-ocannl-926\))$/) {
        next;
      }
      die_malformed("malformed skip report $path:$lno: unexpected trailing line");
    }
  }

  die_malformed("malformed skip report $path: missing backend result") unless defined $b_result;
  if ($b_result eq "FAIL" || $b_result eq "POTENTIAL") {
    die_malformed("malformed skip report $path: backend finding count mismatch (expected $b_expected, got $b_count)")
      unless $b_count == $b_expected;
  }
  if ($seen_comp_e) {
    die_malformed("malformed skip report $path: incomplete environment section") unless defined $e_result;
    if ($e_result eq "FAIL" || $e_result eq "POTENTIAL") {
      die_malformed("malformed skip report $path: environment finding count mismatch (expected $e_expected, got $e_count)")
        unless $e_count == $e_expected;
    }
  }

  return {
    b_status => $b_status,
    b_fail   => \%b_fail,
    b_pot    => \%b_pot,
    e_status => $e_status, # undef for legacy pre-environment reports
    e_fail   => \%e_fail,
    e_pot    => \%e_pot,
  };
}

sub parse_bypass_report {
  my ($path) = @_;
  my $lines = read_lines($path);
  @$lines > 0 or die_malformed("malformed bypass report $path: empty file");

  my $phase = "header";
  my $seen_header_for = 0;
  my $units_none = 0;
  my %declared_units;
  my @declared_order;
  my %seen_units;
  my %unit_state;  # unit -> { kind => "hit"|"cleared"|"inconclusive", detail => ... }
  my $any_reached = 0;
  my $cur_unit;
  my $cur_expected_b = 0;
  my $cur_sum_b = 0;
  my ($res_kind, $u_expected, $u_count) = (undef, 0, 0);
  my %unverified; # claim -> reach_list

  my $finish_unit = sub {
    my ($ctx_lno) = @_;
    return unless defined $cur_unit;
    if ($cur_sum_b != $cur_expected_b) {
      die_malformed("malformed bypass report $path:$ctx_lno: unit $cur_unit bypassed total mismatch (expected $cur_expected_b, got $cur_sum_b)");
    }
    $cur_unit = undef;
    $cur_expected_b = 0;
    $cur_sum_b = 0;
  };

  my $lno = 0;
  for my $line (@$lines) {
    $lno++;
    if ($phase eq "header") {
      if ($line =~ /^bypass rates for [0-9a-f]+ \((?:forced|incremental) execution\)$/) {
        die_malformed("malformed bypass report $path:$lno: duplicate header") if $seen_header_for;
        $seen_header_for = 1;
        next;
      }
      if ($line eq "units: <none>") {
        $units_none = 1;
        $phase = "result_only";
        next;
      }
      if ($line =~ /^units: (.+)$/) {
        my @parts = split(/, /, $1, -1);
        @parts > 0 or die_malformed("malformed bypass report $path:$lno: empty units list");
        for my $u (@parts) {
          $u ne "" && $u !~ /[ \t:]/ or die_malformed("malformed bypass report $path:$lno: invalid unit '\''$u'\''");
          die_malformed("malformed bypass report $path:$lno: duplicate unit '\''$u'\'' in units line")
            if $declared_units{$u}++;
          push @declared_order, $u;
        }
        $phase = "units";
        next;
      }
      die_malformed("malformed bypass report $path:$lno: unexpected line before units");
    }

    if ($phase eq "result_only") {
      if ($line eq "result: NOT AGGREGATED -- no unit'\''s verdict records were supplied") {
        $res_kind = "NOT_AGGREGATED";
        $phase = "done";
        next;
      }
      die_malformed("malformed bypass report $path:$lno: expected NOT AGGREGATED after units: <none>");
    }

    if ($phase eq "units") {
      if ($line =~ /^  bypassed ([0-9]+) of ([0-9]+) \(([0-9]+)%\): (.+): (.+) \[(.+)\]$/) {
        defined $cur_unit && $cur_expected_b > 0
          or die_malformed("malformed bypass report $path:$lno: detail line outside bypassed unit");
        my ($b, $r, $p) = (int($1), int($2), int($3));
        $b >= 1 && $b <= $r or die_malformed("malformed bypass report $path:$lno: invalid bypass counts");
        my $want_p = int((200 * $b + $r) / (2 * $r));
        $p == $want_p or die_malformed("malformed bypass report $path:$lno: inconsistent percent $p% (expected $want_p%)");
        $cur_sum_b += $b;
        next;
      }

      if ($line =~ /^result: (.+)$/) {
        $finish_unit->($lno);
        for my $u (@declared_order) {
          die_malformed("malformed bypass report $path:$lno: missing summary for declared unit '\''$u'\''")
            unless $seen_units{$u};
        }
        my $rest = $1;
        if ($rest eq "CLEAR -- no hatched claim was reached") {
          die_malformed("malformed bypass report $path:$lno: CLEAR no-reached contradicts unit reaches") if $any_reached;
          $res_kind = "CLEAR_NONE";
          $phase = "done";
          next;
        }
        if ($rest eq "CLEAR -- every hatched claim reached was checked on some unit") {
          die_malformed("malformed bypass report $path:$lno: CLEAR checked contradicts zero unit reaches") unless $any_reached;
          $res_kind = "CLEAR_CHECKED";
          $phase = "done";
          next;
        }
        if ($rest =~ /^UNVERIFIED -- ([0-9]+) hatched claim\(s\) bypassed on every unit that reached them$/) {
          die_malformed("malformed bypass report $path:$lno: UNVERIFIED contradicts zero unit reaches") unless $any_reached;
          die_malformed("malformed bypass report $path:$lno: UNVERIFIED count must be >= 1") unless $1 >= 1;
          $res_kind = "UNVERIFIED";
          $u_expected = int($1);
          $phase = "unverified";
          next;
        }
        die_malformed("malformed bypass report $path:$lno: unexpected result line");
      }

      if ($line =~ /^([^ \t][^:]*): no hatched claim reached$/) {
        $finish_unit->($lno);
        my $u = $1;
        $declared_units{$u} or die_malformed("malformed bypass report $path:$lno: undeclared unit '\''$u'\''");
        die_malformed("malformed bypass report $path:$lno: duplicate unit '\''$u'\''") if $seen_units{$u}++;
        $unit_state{$u} = { kind => "inconclusive" };
        $cur_unit = $u;
        $cur_expected_b = 0;
        $cur_sum_b = 0;
        next;
      }

      if ($line =~ /^([^ \t][^:]*): ([0-9]+) of ([0-9]+) hatched claim reach\(es\) bypassed \(([0-9]+)%\), ([0-9]+) checked$/) {
        $finish_unit->($lno);
        my ($u, $b, $r, $p, $c) = ($1, int($2), int($3), int($4), int($5));
        $declared_units{$u} or die_malformed("malformed bypass report $path:$lno: undeclared unit '\''$u'\''");
        die_malformed("malformed bypass report $path:$lno: duplicate unit '\''$u'\''") if $seen_units{$u}++;
        $r >= 1 && $b + $c == $r or die_malformed("malformed bypass report $path:$lno: inconsistent unit counts for '\''$u'\''");
        my $want_p = int((200 * $b + $r) / (2 * $r));
        $p == $want_p or die_malformed("malformed bypass report $path:$lno: inconsistent percent $p% (expected $want_p%)");
        $any_reached = 1;
        if ($b == $r && $c == 0) {
          $unit_state{$u} = { kind => "hit", detail => "$b of $r bypassed, 0 checked" };
        } else {
          $unit_state{$u} = { kind => "cleared" };
        }
        $cur_unit = $u;
        $cur_expected_b = $b;
        $cur_sum_b = 0;
        next;
      }

      die_malformed("malformed bypass report $path:$lno: unexpected line in units section");
    }

    if ($phase eq "unverified") {
      if ($line =~ /^UNVERIFIED: bypassed on every unit that reached it: (.+) \(([^()]+)\)$/) {
        my ($claim, $reaches) = ($1, $2);
        $claim =~ /^.+: .+$/ or die_malformed("malformed bypass report $path:$lno: UNVERIFIED claim missing executable separator");
        my @items = split(/, /, $reaches, -1);
        @items > 0 or die_malformed("malformed bypass report $path:$lno: empty UNVERIFIED unit list");
        for my $item (@items) {
          $item =~ /^([^ \t:]+) ([0-9]+) of ([0-9]+)$/
            or die_malformed("malformed bypass report $path:$lno: malformed UNVERIFIED reach '\''$item'\''");
          my ($ru, $rb, $rr) = ($1, int($2), int($3));
          $declared_units{$ru} or die_malformed("malformed bypass report $path:$lno: UNVERIFIED names undeclared unit '\''$ru'\''");
          $rb >= 1 && $rb == $rr or die_malformed("malformed bypass report $path:$lno: UNVERIFIED reach must be all bypassed");
        }
        die_malformed("malformed bypass report $path:$lno: duplicate UNVERIFIED claim") if exists $unverified{$claim};
        $unverified{$claim} = $reaches;
        $u_count++;
        next;
      }
      die_malformed("malformed bypass report $path:$lno: unexpected line after result: UNVERIFIED");
    }

    if ($phase eq "done") {
      die_malformed("malformed bypass report $path:$lno: unexpected trailing line after result");
    }
  }

  die_malformed("malformed bypass report $path: missing result line") unless defined $res_kind;
  if ($res_kind eq "UNVERIFIED") {
    die_malformed("malformed bypass report $path: UNVERIFIED count mismatch (expected $u_expected, got $u_count)")
      unless $u_count == $u_expected;
  }

  return {
    res_kind   => $res_kind,
    units      => \%unit_state,
    unverified => \%unverified,
  };
}

sub parse_family {
  my ($items_ref, $parser) = @_;
  my @parsed;
  # When --up-to STAMP is set, validate the report matching STAMP first (exit 2
  # if malformed) so a broken current-run report always fails fatally even when
  # an older saved report is also malformed (exit 3).
  if (defined $up_to) {
    for my $item (@$items_ref) {
      next unless defined $item->{stamp} && $item->{stamp} eq $up_to;
      local $MALFORMED_EXIT = 2;
      local $MALFORMED_KIND = "";
      $item->{parsed} = $parser->($item->{path});
    }
  }
  for my $item (@$items_ref) {
    if (exists $item->{parsed}) {
      push @parsed, $item->{parsed};
    } else {
      my $is_old = defined $up_to && defined $item->{stamp} && $item->{stamp} lt $up_to;
      local $MALFORMED_EXIT = $is_old ? 3 : 2;
      local $MALFORMED_KIND = $is_old ? "historical" : "";
      push @parsed, $parser->($item->{path});
    }
  }
  return @parsed;
}

my @parsed_skips  = parse_family(\@skip_items, \&parse_skip_report);
my @parsed_bypass = parse_family(\@bypass_items, \&parse_bypass_report);

# Compute trailing streaks for skip-coverage dimensions.
my (%box_streak, %box_latest, %backend_streak, %backend_latest);
for my $idx (0 .. $#parsed_skips) {
  my $r = $parsed_skips[$idx];
  my $is_last = ($idx == $#parsed_skips);

  # Box (environment) dimension.
  if (defined $r->{e_status}) {
    if ($r->{e_status} eq "complete") {
      my %next_streak;
      for my $c (keys %{$r->{e_fail}}) {
        $next_streak{$c} = ($box_streak{$c} // 0) + 1;
        $box_latest{$c} = 1 if $is_last;
      }
      %box_streak = %next_streak;
    } elsif ($r->{e_status} eq "partial") {
      # Any claim NOT in POTENTIAL was evaluated on some completed box -> cleared.
      for my $c (keys %box_streak) {
        delete $box_streak{$c} unless $r->{e_pot}{$c};
      }
    }
  }

  # Backend dimension.
  if ($r->{b_status} eq "complete") {
    my %next_streak;
    for my $c (keys %{$r->{b_fail}}) {
      $next_streak{$c} = ($backend_streak{$c} // 0) + 1;
      $backend_latest{$c} = 1 if $is_last;
    }
    %backend_streak = %next_streak;
  } elsif ($r->{b_status} eq "partial") {
    for my $c (keys %backend_streak) {
      delete $backend_streak{$c} unless $r->{b_pot}{$c};
    }
  }
}

# Compute trailing streaks for bypass-rate dimensions.
my (%unit_streak, %unit_latest_detail, %unver_streak, %unver_latest_units);
for my $idx (0 .. $#parsed_bypass) {
  my $r = $parsed_bypass[$idx];
  my $is_last = ($idx == $#parsed_bypass);

  # Per-unit 100% bypass dimension.
  for my $u (keys %{$r->{units}}) {
    my $st = $r->{units}{$u};
    if ($st->{kind} eq "hit") {
      $unit_streak{$u} = ($unit_streak{$u} // 0) + 1;
      $unit_latest_detail{$u} = $st->{detail} if $is_last;
    } elsif ($st->{kind} eq "cleared") {
      delete $unit_streak{$u};
    }
  }

  # Cross-unit UNVERIFIED claim dimension.
  if ($r->{res_kind} eq "UNVERIFIED") {
    my %next_streak;
    for my $c (keys %{$r->{unverified}}) {
      $next_streak{$c} = ($unver_streak{$c} // 0) + 1;
      $unver_latest_units{$c} = $r->{unverified}{$c} if $is_last;
    }
    %unver_streak = %next_streak;
  } elsif ($r->{res_kind} eq "CLEAR_CHECKED") {
    %unver_streak = ();
  }
}

my @alerts;
for my $c (sort keys %box_latest) {
  my $n = $box_streak{$c} // 0;
  push @alerts, "STREAK: skipped on every box ($n consecutive sweeps): $c" if $n >= $min_streak;
}
for my $c (sort keys %backend_latest) {
  my $n = $backend_streak{$c} // 0;
  push @alerts, "STREAK: skipped on every known backend ($n consecutive sweeps): $c" if $n >= $min_streak;
}
for my $u (sort keys %unit_latest_detail) {
  my $n = $unit_streak{$u} // 0;
  push @alerts, "STREAK: unit at 100% bypass ($n consecutive sweeps): $u ($unit_latest_detail{$u})" if $n >= $min_streak;
}
for my $c (sort keys %unver_latest_units) {
  my $n = $unver_streak{$c} // 0;
  push @alerts, "STREAK: UNVERIFIED bypassed claim ($n consecutive sweeps): $c ($unver_latest_units{$c})" if $n >= $min_streak;
}

write_line("min-streak: $min_streak");
write_line("skip reports: " . scalar(@skip_reports));
write_line("bypass reports: " . scalar(@bypass_reports));

my $rc = 0;
if (@skip_reports == 0 && @bypass_reports == 0) {
  write_line("result: NOT CHECKED -- no skip or bypass reports supplied");
} elsif (@alerts == 0) {
  write_line("result: CLEAR -- no streak reached $min_streak consecutive sweeps");
} else {
  write_line("result: ALERT -- " . scalar(@alerts) . " streak(s) reached $min_streak consecutive sweeps");
  for my $a (@alerts) {
    write_line($a);
  }
  $rc = 1;
}

close(STDOUT) or die_input("cannot write report");
exit $rc;
' -- "$@"
