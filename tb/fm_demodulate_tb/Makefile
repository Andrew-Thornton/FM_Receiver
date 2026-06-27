# Makefile

# defaults
SIM ?= nvc
TOPLEVEL_LANG ?= vhdl

VHDL_LIB_ORDER = vhdl_common

VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/math_utils_pkg.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/complex_mult.vhd
VHDL_SOURCES += $(PWD)/../../src/fm_demodulate.vhd

COCOTB_TOPLEVEL = fm_demodulate

COCOTB_TEST_MODULES = fm_demodulate_tb

SIM_ARGS += --wave=wave.vcd
SIM_ARGS += --dump-arrays

include $(shell cocotb-config --makefiles)/Makefile.sim
