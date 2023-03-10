data work.data;
	set sassym.all_lags_oecd;
	EG_ELC_coal_ZS_ch_lag3 = lag3(EG_ELC_coal_ZS_change);
run;

proc sort data=work.data;
	BY COUNTRYNAME YEAR;
RUN;	

PROC PANEL DATA=work.data PLOTS=all;
	WHERE year > 1994 and year < 2016 and countryname not in ("Turkey","Luxembourg");
	MODEL EFPerGDP = EG_USE_COMM_GD_PP_KD
					SE_XPD_TOTL_GB_ZS
					SP_DYN_LE00_IN
					VC_IHR_PSRC_P5
					SP_URB_TOTL_IN_ZS
					SP_POP_GROW
					EG_ELC_coal_ZS
					EG_ELC_NUCL_ZS
					EG_ELC_HYRO_ZS
					EG_ELC_RNWX_ZS
					/fixtwo printfixed;
	id CountryName Year;
run;

data work.data2;
	set work.data;
	keep CountryName Year
			EFPerGDP
		EG_USE_COMM_GD_PP_KD
					SE_XPD_TOTL_GB_ZS
					SP_DYN_LE00_IN
					VC_IHR_PSRC_P5
					SP_URB_TOTL_IN_ZS
					SP_POP_GROW
					EG_ELC_coal_ZS
					EG_ELC_NUCL_ZS
					EG_ELC_HYRO_ZS
					EG_ELC_RNWX_ZS
					GDPPerCap;
run;


