-------------------------------------------------------------------------------
-- Author        : Andrew Thornton
-- Standard      : VHDL 2008
-------------------------------------------------------------------------------
-- Description   : a fun fm receiver project
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

library vhdl_common;
use vhdl_common.math_utils_pkg.ceil_log2;

entity fm_demodulate is
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
end fm_demodulate;

architecture rtl of fm_demodulate is

  signal real_conj_z : signed(INPUT_DATA_W-1 downto 0) := (others => '0');
  signal imag_conj_z : signed(INPUT_DATA_W-1 downto 0) := (others => '0');

  signal real_mult_with_prev_conj : signed(2*INPUT_DATA_W downto 0) := (others => '0');
  signal imag_mult_with_prev_conj : signed(2*INPUT_DATA_W downto 0) := (others => '0');
  signal mult_vld : std_logic := '0';

begin

  process(clk_i)
  begin
    if rising_edge(clk_i) then
      if vld_i = '1' then 
        real_conj_z <= real_i;
        imag_conj_z <= 0-imag_i;
      end if;
    end if;
  end process;

  complex_mult : entity vhdl_common.complex_mult
  port map(
    clk_i    => clk_i,
    srst_i   => srst_i,
    a_real_i => real_i,
    a_imag_i => imag_i,
    b_real_i => real_conj_z,
    b_imag_i => imag_conj_z,
    vld_i    => vld_i,
    c_real_o => real_mult_with_prev_conj,
    c_imag_o => imag_mult_with_prev_conj,
    vld_o    => mult_vld
  );




  -- atan2_cordic


end rtl;
