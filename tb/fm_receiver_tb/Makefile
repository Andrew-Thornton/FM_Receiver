# Makefile

# defaults
SIM ?= nvc
TOPLEVEL_LANG ?= vhdl

VHDL_LIB_ORDER = vhdl_common

VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/math_utils_pkg.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/integrator_stage.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/integrator.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/raw_decimator.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/comb_filter_stage.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/comb_filter.vhd
VHDL_SOURCES_vhdl_common += $(PWD)/../../sub/VHDL_Common/src/dsp/dec_cic_filter.vhd
VHDL_SOURCES += $(PWD)/../../src/fm_receiver.vhd

COCOTB_TOPLEVEL = fm_receiver

COCOTB_TEST_MODULES = fm_receiver_tb

SIM_ARGS += --wave=wave.vcd
SIM_ARGS += --dump-arrays

include $(shell cocotb-config --makefiles)/Makefile.sim
