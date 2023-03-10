%LET _CLIENTTASKLABEL='Merge Footprint and GDP';
%LET _CLIENTPROCESSFLOWNAME='Process Flow';
%LET _CLIENTPROJECTPATH='C:\Users\Neil\Desktop\Sas2020_CombiningFootprintGDP.egp';
%LET _CLIENTPROJECTPATHHOST='HOME';
%LET _CLIENTPROJECTNAME='Sas2020_CombiningFootprintGDP.egp';
%LET _SASPROGRAMFILE='';
%LET _SASPROGRAMFILEHOST='';

GOPTIONS ACCESSIBLE;
proc sort data=work.worldgdpdata
		  out=work.worldgdpdata;
		  by CountryName;
run;

proc transpose 	data=work.worldgdpdata
				out=work.Worldgdp_Trans
				name=year;
			by CountryName;
			id indicatorName;
run;
proc sort data=work.Footprint
		  out=work.Footprint;
		  by Country;
run;

proc transpose 	data=work.Footprint
				out=work.Footprint_Trans
				name=year;
			var total;
			by Country year;
			id record;
	run;

data work.worldgdp_trans;
	set work.worldgdp_trans;
	Yr = input(year, 4.);
	drop year;
	rename yr = Year;
	run;

data work.footprint_trans;
	set work.footprint_trans;
	rename country = CountryName;
run;

data work.GDP_Footprint;
	Merge work.footprint_trans work.worldgdp_trans;
	by CountryName year;
run;

GOPTIONS NOACCESSIBLE;
%LET _CLIENTTASKLABEL=;
%LET _CLIENTPROCESSFLOWNAME=;
%LET _CLIENTPROJECTPATH=;
%LET _CLIENTPROJECTPATHHOST=;
%LET _CLIENTPROJECTNAME=;
%LET _SASPROGRAMFILE=;
%LET _SASPROGRAMFILEHOST=;

