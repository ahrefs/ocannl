#!/usr/bin/env bash
# Shared advisory report for read-selected golden families (gh-ocannl-2073).
# Sourcing only defines the fail-closed Dune reader and the report function.
# The reader is also used by test-run.sh to resolve logical alias directories.
dune_reader_perl='
  use strict; use warnings;
  sub read_dune {
    my ($f) = @_;
    open my $fh, "<", $f or die "cannot read $f";
    local $/;
    my $t = <$fh> // "";
    close $fh;
    my (@tok, $ok);
    $ok = eval {
      pos($t) = 0;
      while (pos($t) < length $t) {
        if ($t =~ /\G\s+/gc || $t =~ /\G;[^\n]*/gc) { next }
        # `#|` and `#;` are where dune readings differ (comments in some
        # versions, atom text in others): a file holding one is not read.
        die if $t =~ /\G#[|;]/gc;
        if ($t =~ /\G([()])/gc) { push @tok, [$1]; next }
        if ($t =~ /\G"((?:[^"\\]|\\.)*)"/gcs) { push @tok, ["a", $1]; next }
        if ($t =~ /\G([^\s();"]+)/gc) { push @tok, ["a", $1]; next }
        die;
      }
      1;
    };
    die "unsupported Dune syntax" unless $ok;
    # Into nested lists.
    my @stack = ([]);
    for my $k (@tok) {
      if ($k->[0] eq "(") { push @stack, []; next }
      my $x;
      if ($k->[0] eq ")") { @stack > 1 or die "unbalanced Dune form"; $x = pop @stack }
      else { $x = $k->[1] }
      push @{ $stack[-1] }, $x;
    }
    die "unbalanced Dune form" if @stack != 1;
    my $check;
    $check = sub {
      my ($form) = @_;
      return unless ref $form;
      die "included Dune stanzas" if @$form && !ref $form->[0] && $form->[0] =~ /^(?:include|dynamic_include)$/;
      $check->($_) for @$form;
    };
    $check->($stack[0]);
    return $stack[0];
  }
'

golden_families_perl='
  my ($mode) = @ARGV;
  my %selected;
  while (<STDIN>) {
    chomp;
    next if m{^(?:/|\\|[A-Za-z]:|\.\./)} || m{/\.\./};
    $selected{$_} = 1 if /\.expected$/ && -f $_;
  }
  my (%by_dir, %reported);
  for my $file (sort keys %selected) {
    my ($dir, $base) = $file =~ m{^(?:(.*)/)?([^/]+)$};
    $dir = "." unless defined $dir;
    push @{ $by_dir{$dir} }, $base;
  }
  for my $dir (sort keys %by_dir) {
    my $file = -f "$dir/dune" ? "$dir/dune" : "$dir/dune-file";
    my $forms = eval { read_dune($file) };
    unless ($forms) {
      print "golden families: family report unavailable for $dir (unreadable or unsupported Dune file)\n";
      next;
    }
    my $walk;
    $walk = sub {
      my ($form) = @_;
      return unless ref $form;
      # Rules in logical subdirectories do not belong to the local filenames.
      return if @$form && !ref $form->[0] && $form->[0] eq "subdir";
      if (@$form >= 2 && !ref $form->[0] && $form->[0] =~ /^diff\??$/ && !ref $form->[1]) {
        my $name = $form->[1];
        if ($name =~ /^(.*?)%\{read:[^}]+\}(.*\.expected)$/) {
          my ($prefix, $suffix) = ($1, $2);
          # One read interpolation, local literal prefix and suffix only.
          unless ("$prefix$suffix" =~ m{[/\\%]}) {
            my $pattern = qr/^\Q$prefix\E[^\/]+\Q$suffix\E$/;
            if (grep { /$pattern/ } @{ $by_dir{$dir} }) {
              my $family = "$dir/$name";
              unless ($reported{$family}++) {
                opendir my $dh, $dir or die "cannot list $dir: $!\n";
                my @members = sort grep { /$pattern/ && -f "$dir/$_" } readdir $dh;
                closedir $dh;
                print "golden families: $family\n";
                for my $member (@members) {
                  my $path = $dir eq "." ? $member : "$dir/$member";
                  my $state = $selected{$path}
                    ? ($mode eq "pending" ? "pending promotion" : "promoted")
                    : ($mode eq "pending" ? "NOT pending promotion" : "NOT promoted");
                  print "  $state: $path\n";
                }
              }
            }
          }
        }
      }
      $walk->($_) for @$form;
    };
    $walk->($forms);
  }
'
golden_family_report() { # ROOT pending|applied; selected paths on stdin
  (
    [ -n "$1" ] && cd -- "$1" || { echo "golden families: family report unavailable (worktree missing)"; exit 0; }
    perl -e "$dune_reader_perl $golden_families_perl" "$2" ||
      echo "golden families: WARNING -- family report failed"
  )
}
