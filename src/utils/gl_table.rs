use crate::utils::error_codes::ErrorCode;

pub const MAX_GL_ORDER: usize = 15;

pub fn get_gl_weights_and_abscissae(order: usize, index: usize) -> Result<(f64, f64), ErrorCode>
{
    let ref_abs: f64 = match order 
    {
        1 => LEGENDRE_ABSCISSA_1[index],
        2 => LEGENDRE_ABSCISSA_2[index],
        3 => LEGENDRE_ABSCISSA_3[index],
        4 => LEGENDRE_ABSCISSA_4[index],
        5 => LEGENDRE_ABSCISSA_5[index],
        6 => LEGENDRE_ABSCISSA_6[index],
        7 => LEGENDRE_ABSCISSA_7[index],
        8 => LEGENDRE_ABSCISSA_8[index],
        9 => LEGENDRE_ABSCISSA_9[index],
        10 => LEGENDRE_ABSCISSA_10[index],
        11 => LEGENDRE_ABSCISSA_11[index],
        12 => LEGENDRE_ABSCISSA_12[index],
        13 => LEGENDRE_ABSCISSA_13[index],
        14 => LEGENDRE_ABSCISSA_14[index],
        15 => LEGENDRE_ABSCISSA_15[index],
        _ => return Err(ErrorCode::GaussianQuadratureOrderOutOfRange),
    };

    let ref_weight: f64 = match order 
    {
        1 => LEGENDRE_WEIGHT_1[index],
        2 => LEGENDRE_WEIGHT_2[index],
        3 => LEGENDRE_WEIGHT_3[index],
        4 => LEGENDRE_WEIGHT_4[index],
        5 => LEGENDRE_WEIGHT_5[index],
        6 => LEGENDRE_WEIGHT_6[index],
        7 => LEGENDRE_WEIGHT_7[index],
        8 => LEGENDRE_WEIGHT_8[index],
        9 => LEGENDRE_WEIGHT_9[index],
        10 => LEGENDRE_WEIGHT_10[index],
        11 => LEGENDRE_WEIGHT_11[index],
        12 => LEGENDRE_WEIGHT_12[index],
        13 => LEGENDRE_WEIGHT_13[index],
        14 => LEGENDRE_WEIGHT_14[index],
        15 => LEGENDRE_WEIGHT_15[index],
        _ => return Err(ErrorCode::GaussianQuadratureOrderOutOfRange),
    };

    return Ok((ref_abs, ref_weight));
}


// =============================================================================
// Table taken from https://pomax.github.io/bezierinfo/legendre-gauss.html
// =============================================================================
const LEGENDRE_ABSCISSA_1: [f64; 1] = [0.0];
const LEGENDRE_ABSCISSA_2: [f64; 2] = [-0.577_350_269_189_626, 0.577_350_269_189_626];
const LEGENDRE_ABSCISSA_3: [f64; 3] = [0.0, -0.774_596_669_241_483, 0.774_596_669_241_483];
const LEGENDRE_ABSCISSA_4: [f64; 4] = [-0.339_981_043_584_856, 0.339_981_043_584_856, -0.861_136_311_594_053, 0.861_136_311_594_053];
const LEGENDRE_ABSCISSA_5: [f64; 5] = 
[
    0.0,
    -0.538_469_310_105_683,
    0.538_469_310_105_683,
    -0.906_179_845_938_664,
    0.906_179_845_938_664
];
const LEGENDRE_ABSCISSA_6: [f64; 6] = 
[
    0.661_209_386_466_265,
    -0.661_209_386_466_265,
    -0.238_619_186_083_197, 
    0.238_619_186_083_197,
    -0.932_469_514_203_152,
    0.932_469_514_203_152
];
const LEGENDRE_ABSCISSA_7: [f64; 7] = 
[
    0.0,
    0.405_845_151_377_397,
    -0.405_845_151_377_397,
    -0.741_531_185_599_394,
    0.741_531_185_599_394,
    -0.949_107_912_342_759,
    0.949_107_912_342_759,
];
const LEGENDRE_ABSCISSA_8: [f64; 8] = 
[
    -0.183_434_642_495_65,
    0.183_434_642_495_65,
    -0.525_532_409_916_329,
    0.525_532_409_916_329,
    -0.796_666_477_413_627,
    0.796_666_477_413_627,
    -0.960_289_856_497_536,
    0.960_289_856_497_536,
];
const LEGENDRE_ABSCISSA_9: [f64; 9] = 
[
    0.0,
    -0.836_031_107_326_636,
    0.836_031_107_326_636,
    -0.968_160_239_507_626,
    0.968_160_239_507_626,
    -0.324_253_423_403_809,
    0.324_253_423_403_809,
    -0.613_371_432_700_59,
    0.613_371_432_700_59,
];
const LEGENDRE_ABSCISSA_10: [f64; 10] = 
[
    -0.148_874_338_981_631,
    0.148_874_338_981_631,
    -0.433_395_394_129_247,
    0.433_395_394_129_247,
    -0.679_409_568_299_024,
    0.679_409_568_299_024,
    -0.865_063_366_688_985,
    0.865_063_366_688_985,
    -0.973_906_528_517_172,
    0.973_906_528_517_172,
];
const LEGENDRE_ABSCISSA_11: [f64; 11] = 
[
    0.0,
    -0.269_543_155_952_345,
    0.269_543_155_952_345,
    -0.519_096_129_110_681,
    0.519_096_129_110_681,
    -0.730_152_005_574_049,
    0.730_152_005_574_049,
    -0.887_062_599_768_095,
    0.887_062_599_768_095,
    -0.978_228_658_146_057,
    0.978_228_658_146_057,
];
const LEGENDRE_ABSCISSA_12: [f64; 12] = 
[
    -0.125_333_408_511_469,
    0.125_333_408_511_469,
    -0.367_831_498_918_18,
    0.367_831_498_918_18,
    -0.587_317_954_286_617,
    0.587_317_954_286_617,
    -0.769_902_674_194_305,
    0.769_902_674_194_305,
    -0.904_117_256_370_475,
    0.904_117_256_370_475,
    -0.981_560_634_246_719,
    0.981_560_634_246_719,
];
const LEGENDRE_ABSCISSA_13: [f64; 13] = 
[
    0.0,
    -0.230_458_315_955_135,
    0.230_458_315_955_135,
    -0.448_492_751_036_447,
    0.448_492_751_036_447,
    -0.642_349_339_440_34,
    0.642_349_339_440_34,
    -0.801_578_090_733_31,
    0.801_578_090_733_31,
    -0.917_598_399_222_978,
    0.917_598_399_222_978,
    -0.984_183_054_718_588,
    0.984_183_054_718_588,
];
const LEGENDRE_ABSCISSA_14: [f64; 14] = 
[
    -0.108_054_948_707_344,
    0.108_054_948_707_344,
    -0.319_112_368_927_89,
    0.319_112_368_927_89,
    -0.515_248_636_358_154,
    0.515_248_636_358_154,
    -0.687_292_904_811_685,
    0.687_292_904_811_685,
    -0.827_201_315_069_765,
    0.827_201_315_069_765,
    -0.928_434_883_663_574,
    0.928_434_883_663_574,
    -0.986_283_808_696_812,
    0.986_283_808_696_812,
];
const LEGENDRE_ABSCISSA_15: [f64; 15] = 
[
    0.0,
    -0.201_194_093_997_435,
    0.201_194_093_997_435,
    -0.394_151_347_077_563,
    0.394_151_347_077_563,
    -0.570_972_172_608_539,
    0.570_972_172_608_539,
    -0.724_417_731_360_17,
    0.724_417_731_360_17,
    -0.848_206_583_410_427,
    0.848_206_583_410_427,
    -0.937_273_392_400_706,
    0.937_273_392_400_706,
    -0.987_992_518_020_485,
    0.987_992_518_020_485,
];


const LEGENDRE_WEIGHT_1: [f64; 1] = [2.0];
const LEGENDRE_WEIGHT_2: [f64; 2] = [1.0, 1.0];
const LEGENDRE_WEIGHT_3: [f64; 3] = [0.888_888_888_888_889, 0.555_555_555_555_555_6, 0.555_555_555_555_555_6];
const LEGENDRE_WEIGHT_4: [f64; 4] = 
[
    0.652_145_154_862_546, 
    0.652_145_154_862_546, 
    0.347_854_845_137_454, 
    0.347_854_845_137_454
];
const LEGENDRE_WEIGHT_5: [f64; 5] = 
[
    0.568_888_888_888_889, 
    0.478_628_670_499_366, 
    0.478_628_670_499_366,
    0.236_926_885_056_189_1,
    0.236_926_885_056_189_1
];
const LEGENDRE_WEIGHT_6: [f64; 6] = 
[
    0.360_761_573_048_139,
    0.360_761_573_048_139,
    0.467_913_934_572_691, 
    0.467_913_934_572_691,
    0.171_324_492_379_17,
    0.171_324_492_379_17
];
const LEGENDRE_WEIGHT_7: [f64; 7] =
[
    0.417_959_183_673_469,
    0.381_830_050_505_119,
    0.381_830_050_505_119,
    0.279_705_391_489_277,
    0.279_705_391_489_277,
    0.129_484_966_168_87,
    0.129_484_966_168_87,
];
const LEGENDRE_WEIGHT_8: [f64; 8] =
[
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887,
    0.313_706_645_877_887,
    0.222_381_034_453_374,
    0.222_381_034_453_374,
    0.101_228_536_290_376,
    0.101_228_536_290_376,
];
const LEGENDRE_WEIGHT_9: [f64; 9] =
[
    0.330_239_355_001_26,
    0.180_648_160_694_857,
    0.180_648_160_694_857,
    0.081_274_388_361_574,
    0.081_274_388_361_574,
    0.312_347_077_040_003,
    0.312_347_077_040_003,
    0.260_610_696_402_935,
    0.260_610_696_402_935,    
];
const LEGENDRE_WEIGHT_10: [f64; 10] =
[
    0.295_524_224_714_753,
    0.295_524_224_714_753,
    0.269_266_719_309_996,
    0.269_266_719_309_996,
    0.219_086_362_515_982,
    0.219_086_362_515_982,
    0.149_451_349_150_581,
    0.149_451_349_150_581,
    0.066_671_344_308_688,
    0.066_671_344_308_688,
];
const LEGENDRE_WEIGHT_11: [f64; 11] =
[
    0.272_925_086_777_901,
    0.262_804_544_510_247,
    0.262_804_544_510_247,
    0.233_193_764_591_99,
    0.233_193_764_591_99,
    0.186_290_210_927_734,
    0.186_290_210_927_734,
    0.125_580_369_464_905,
    0.125_580_369_464_905,
    0.055_668_567_116_174,
    0.055_668_567_116_174,
];
const LEGENDRE_WEIGHT_12: [f64; 12] =
[
    0.249_147_045_813_403,
    0.249_147_045_813_403,
    0.233_492_536_538_355,
    0.233_492_536_538_355,
    0.203_167_426_723_066,
    0.203_167_426_723_066,
    0.160_078_328_543_346,
    0.160_078_328_543_346,
    0.106_939_325_995_318,
    0.106_939_325_995_318,
    0.047_175_336_386_512,
    0.047_175_336_386_512,
];
const LEGENDRE_WEIGHT_13: [f64; 13] =
[
    0.232_551_553_230_874,
    0.226_283_180_262_897,
    0.226_283_180_262_897,
    0.207_816_047_536_889,
    0.207_816_047_536_889,
    0.178_145_980_761_946,
    0.178_145_980_761_946,
    0.138_873_510_219_787,
    0.138_873_510_219_787,
    0.092_121_499_837_728,
    0.092_121_499_837_728,
    0.040_484_004_765_316,
    0.040_484_004_765_316,
];
const LEGENDRE_WEIGHT_14: [f64; 14] =
[
    0.215_263_853_463_158,
    0.215_263_853_463_158,
    0.205_198_463_721_29,
    0.205_198_463_721_29,
    0.185_538_397_477_938,
    0.185_538_397_477_938,
    0.157_203_167_158_194,
    0.157_203_167_158_194,
    0.121_518_570_687_903,
    0.121_518_570_687_903,
    0.080_158_087_159_76,
    0.080_158_087_159_76,
    0.035_119_460_331_752,
    0.035_119_460_331_752,
];
const LEGENDRE_WEIGHT_15: [f64; 15] =
[
    0.202_578_241_925_561,
    0.198_431_485_327_111,
    0.198_431_485_327_111,
    0.186_161_000_015_562,
    0.186_161_000_015_562,
    0.166_269_205_816_994,
    0.166_269_205_816_994,
    0.139_570_677_926_154,
    0.139_570_677_926_154,
    0.107_159_220_467_172,
    0.107_159_220_467_172,
    0.070_366_047_488_108,
    0.070_366_047_488_108,
    0.030_753_241_996_117,
    0.030_753_241_996_117,
];