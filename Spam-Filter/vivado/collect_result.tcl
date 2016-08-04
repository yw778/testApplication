#============================================================================
# collect-result.tcl
#============================================================================
# @brief: A Tcl script that collect and dumps out results from csim & synthesis.
# @desc:
# 1. print out header infomation if the result file is newly created
# 2. collect accuracy results from out.dat
# 3. collect synthesis results from Vivado_HLS synthesis report

#---------------------------
# print header information
#---------------------------
set filename [lindex $argv 0]
set hls_prj [lindex $argv 1]
set info "${hls_prj}"
file mkdir "./result"
if { ! [ file exists "./result/${filename}"] } {
  set fileId [open "./result/${filename}" w]
  if { [ file exist "${hls_prj}/solution1/sim/report/SgdLR_cosim.rpt" ] } {
    set msg "Design Accuracy CP BRAM DSP FF LUT Latency cosim-Latency"
  } else {
    set msg "Design Accuracy CP BRAM DSP FF LUT Latency"
  }
  puts $fileId $msg
  close $fileId
}

#---------------------------
# colect accuracy results
#---------------------------
set fileId [open "./result/${filename}" a+]
set info [lindex [split $info "."] 0]
puts -nonewline $fileId "${info}"
file copy -force "${hls_prj}/solution1/csim/build/out.dat" "./result/out_${info}.dat"
set fp [open "./${hls_prj}/solution1/csim/build/out.dat" r]
set file_data [read $fp]
close $fp
set data [split $file_data "\n"]
foreach line $data {
  if { [string match "*Overall Error*" $line] } {
    set info [lindex [split $line "="] 1]
    puts -nonewline $fileId "${info} "
    break
  }
}

#---------------------------
# colect synthesis results
#---------------------------
set fp [open "${hls_prj}/solution1/syn/report/SgdLR_csynth.xml" r]
set file_data [read $fp]
close $fp
set data [split $file_data "\n"]
foreach { pattern } {
  "*EstimatedClockPeriod*"
  "*BRAM_18K*"
  "*DSP48E*"
  "*FF*"
  "*LUT*"
  "*Average-caseLatency*"
} {
foreach line $data {
  if { [string match $pattern $line] } {
    set info [lindex [split [lindex [split $line "<"] 1] ">"] 1]
    puts -nonewline $fileId "${info} "
    break
  }
}
}

#---------------------------------------
# colect cosim latency result if exists
#---------------------------------------
# if cosim latency result exists, then grep from cosim result
if { [ file exist "${hls_prj}/solution1/sim/report/SgdLR_cosim.rpt" ] } {
  set fp [open "${hls_prj}/solution1/sim/report/SgdLR_cosim.rpt" r]
  set file_data [read $fp]
  close $fp
  set data [split $file_data "\n"]
  foreach { pattern } {
    "*Verilog*"
  } {
  foreach line $data {
    if { [string match $pattern $line] } {
      set info [string trim [lindex [split $line "|"] 4]]
      puts -nonewline $fileId "${info} "
      break
    }
  }
  }
}

puts $fileId "\t"
close $fileId
