#!/usr/bin/env perl

use warnings;
use strict;

use Cwd;
use Cwd 'abs_path';

use Digest::SHA qw(sha1_hex);
use File::Copy;
use File::Copy::Recursive qw(dircopy dirmove);
use File::Path qw(make_path remove_tree);
use File::Basename;

use POSIX;

use Math::Complex;

my $phase = "REST"; #REST OR STRESS
my $params_file = "C:\\Users\\MLL\\Desktop\\Rb82\\params-STAT.txt";
my @patients = <"C:\\Users\\MLL\\Desktop\\Rb82\\2016\\*">;

for my $patient (@patients){
    #print("Destionation is $patient \n $params_file \n")
    #system("cscript C:\\JSRecon12\\JSRecon12.js $patient\\$phase $params_file");
    chdir("$patient\\$phase-Converted\\$phase-LM-00");
    system("Run-00-$phase-LM-00-Histogramming.bat");
    system("Run-01-$phase-LM-00-Makeumap.bat");
    #Change offset - Christoffer
    open my $fh_r, "<" , "Run-04-$phase-LM-00-PSFTOF.bat" or die "Could not open: Run-04-$phase-LM-00-PSFTOF.bat \n";
    my @file_lines;
    my $counter = 0;
    while(my $line = <$fh_r>) {
        @file_lines[$counter] = $line;
        if ($counter == 20){
            $counter++;
        } 
                $counter++;
    }
    @file_lines[21] = "set cmd= \%cmd\% --xoffs 30\n";
    close($fh_r);
    open my $fh_w, ">" , "Run-04-$phase-LM-00-PSFTOF.bat" or die;
    foreach(@file_lines){
        print $fh_w "$_";
    }
    close($fh_w);
    system("Run-04-$phase-LM-00-PSFTOF.bat");
}