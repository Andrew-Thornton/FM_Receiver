"""
-------------------------------------------------------------------------------
-- Author        : Andrew Thornton
-- Standard      : cocotb / Python 3
-------------------------------------------------------------------------------
-- Description
--   cocotb testbench for the fm_receiver entity.
--   inputs one second of fm data
--   and the output should be a stereo output which is put into a wav file
-------------------------------------------------------------------------------
"""

import math
import itertools
import numpy as np

import cocotb
from cocotb.clock    import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer

CLOCK_PERIOD_PS = 4068
CLOCK_HOLD_PS   = 400

async def initialise(dut):
    dut.srst_i.value = 0
    dut.real_i.value = 0
    dut.imag_i.value = 0
    dut.vld_i.value  = 0

async def reset(dut):
    dut.srst_i.value = 1

    # Hold reset for several cycles
    await ClockCycles(dut.clk_i, 8)

    # De-assert on a rising edge, then wait for pipeline to drain
    await RisingEdge(dut.clk_i)
    dut.srst_i.value = 0


@cocotb.test()
async def run_fm_demodulate(dut):
    dut._log.info(f"Starting run_fm_demodulate tb")
    dut._log.info(f"Starting clock soon")
    cocotb.start_soon(Clock(dut.clk_i, CLOCK_PERIOD_PS, unit="ps").start())

    dut._log.info(f"Initialising")
    await initialise(dut)
    dut._log.info(f"Initialisation Complete")
    dut._log.info(f"Resetting")
    await reset(dut)
    dut._log.info(f"Reset complete")

    data_input_task = cocotb.start_soon(data_inputter(dut))

    await data_input_task

    await ClockCycles(dut.clk_i, 1000)

    dut._log.info(f"run_fm_demodulate tb complete") 
