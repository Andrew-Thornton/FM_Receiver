-------------------------------------------------------------------------------
-- Author        : Andrew Thornton
-- Standard      : VHDL 2008
-------------------------------------------------------------------------------
-- Description : a fun fm receiver project
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vhdl_common;
use vhdl_common.math_utils_pkg.ceil_log2;

entity fm_receiver is
  generic(
    INPUT_CLK_FREQ_HZ : positive := 245760000;
    INPUT_DATA_W      : positive := 16;
    OUTPUT_DATA_W     : positive := 16
  );
  port(
    clk_i   : in  std_logic;
    srst_i  : in  std_logic;
    real_i  : in  signed(INPUT_DATA_W-1 downto 0);
    imag_i  : in  signed(INPUT_DATA_W-1 downto 0);
    vld_i   : in  std_logic := '1';
    left_o  : out signed(OUTPUT_DATA_W-1 downto 0);
    right_o : out signed(OUTPUT_DATA_W-1 downto 0);
    b_vld_o : out std_logic
  );
end fm_receiver;

architecture rtl of fm_receiver is

  constant DESIRED_OUTPUT_FREQ_HZ : positive := 120000;
  constant DIVISION_REQUIRED : integer := INPUT_CLK_FREQ_HZ/DESIRED_OUTPUT_FREQ_HZ;
  constant NUM_STAGES : integer := ceil_log2(DIVISION_REQUIRED);
  constant DIVISION_REQUIRED_UNS : unsigned(NUM_STAGES downto 0) := to_unsigned(DIVISION_REQUIRED,NUM_STAGES+1);

  type signed_array_t is array (natural range <>) of signed;
  signal real_stage : signed_array_t(0 to NUM_STAGES)(INPUT_DATA_W-1 downto 0) := (others => (others => '0'));
  signal imag_stage : signed_array_t(0 to NUM_STAGES)(INPUT_DATA_W-1 downto 0) := (others => (others => '0'));
  signal stage_vld    : std_logic_vector(0 to NUM_STAGES) := (others => '0');

begin

  assert INPUT_CLK_FREQ_HZ = 245760000
  report "Only currently support 245.76MHz Clock freq (Over Kill I know)"
  severity failure;

  assert INPUT_CLK_FREQ_HZ mod DESIRED_OUTPUT_FREQ_HZ = 0
  report "Input clock frequency is not divisble by 120KHz"
  severity failure;
  
  assert (DIVISION_REQUIRED_UNS > 0) and ((DIVISION_REQUIRED_UNS and (DIVISION_REQUIRED_UNS - 1)) = 0)
  report "DIVISION_REQUIRED must be a power of 2"
  severity failure;

  real_stage(0) <= real_i;
  imag_stage(0) <= imag_i;
  stage_vld(0)  <= vld_i;

  decimation_stage_generator_g : for stage in 0 to NUM_STAGES-1 generate

    my_dec_real_stage_i : entity vhdl_common.dec_cic_filter
    generic map(
      DECIMATION_RATE_R  => 2,
      DIFFERENTIAL_DELAY => 1,
      NUMBER_TAPS_N      => 1,
      INPUT_DATA_W       => real_i'length
    )port map(
      clk_i   => clk_i,
      srst_i  => srst_i,
      a_i     => real_stage(stage),
      a_vld_i => stage_vld(stage),
      b_o     => real_stage(stage+1),
      b_vld_o => stage_vld(stage+1)
    );

    my_dec_imag_stage_i : entity vhdl_common.dec_cic_filter
    generic map(
      DECIMATION_RATE_R  => 2,
      DIFFERENTIAL_DELAY => 1,
      NUMBER_TAPS_N      => 1,
      INPUT_DATA_W       => imag_i'length
    )port map(
      clk_i   => clk_i,
      srst_i  => srst_i,
      a_i     => imag_stage(stage),
      a_vld_i => stage_vld(stage),
      b_o     => imag_stage(stage+1),
      b_vld_o => open
    );

  end generate;
  
-- TODO
--   my_stero_fm_demodulator : entity fm_demodulate
--   generic map(

--   )port map(

--   );

--  To do, I2S Driver

end rtl;
