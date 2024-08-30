use num_complex::Complex;
use rustfft;
use std::sync::Arc;

/// Performs an FFT correlation. Equivalent to scipy.signal.correlate(a, b, mode='same').
///
/// Wraps the `rustfft` planner and transforms to reduce run-time for a constant size input.
pub struct FftCorrelate32 {
    fft_size: usize,
    //planner: rustfft::FftPlanner<f32>,
    forward: Arc<dyn rustfft::Fft<f32>>,
    inverse: Arc<dyn rustfft::Fft<f32>>,
}

impl FftCorrelate32 {
    pub fn new(a_size: usize, b_size: usize) -> FftCorrelate32 {
        let fft_size = a_size + b_size+ 1;
        let mut planner = rustfft::FftPlanner::new();
        let forward = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);
        let inverse = planner.plan_fft(fft_size, rustfft::FftDirection::Inverse);
        FftCorrelate32 {
            fft_size: fft_size,
            //planner: planner,
            forward: forward,
            inverse: inverse,
        }
    }

    pub fn correlate(
        &self, 
        a: &[Complex<f32>],
        b: &[Complex<f32>]) -> Vec<Complex<f32>>
    {
        let mut ap = vec!(
            Complex::<f32> {
                re: 0.0, im: 0.0 
            }; self.fft_size
        );
        
        let mut bp = vec!(
            Complex::<f32> {
                re: 0.0, im: 0.0
            }; self.fft_size
        );

        for (avv, av) in ap.iter_mut().zip(a.iter()) {
            *avv = *av;
        }

        for (bvv, bv) in bp.iter_mut().zip(b.iter()) {
            *bvv = *bv;
        }

        self.forward.process(&mut ap);
        self.forward.process(&mut bp);

        for (av, bv) in ap.iter_mut().zip(bp.iter()) {
            *av = *av * bv.conj();
        }

        self.inverse.process(&mut ap);
        
        let adj = (b.len() - 1) * 2;
        let valid = a.len() - b.len() + 1;

        let mut out: Vec<Complex<f32>> = Vec::with_capacity(
            valid
        );

        for x in 0..valid {
            out.push(
                ap[(adj + x) % ap.len()] / ap.len() as f32
            );
        }
        
        out
    }
}

pub struct FftCorrelate64 {
    fft_size: usize,
    //planner: rustfft::FftPlanner<f64>,
    forward: Arc<dyn rustfft::Fft<f64>>,
    inverse: Arc<dyn rustfft::Fft<f64>>,
}

impl FftCorrelate64 {
    pub fn new(a_size: usize, b_size: usize) -> FftCorrelate64 {
        let fft_size = a_size + b_size+ 1;
        let mut planner = rustfft::FftPlanner::new();
        let forward = planner.plan_fft(fft_size, rustfft::FftDirection::Forward);
        let inverse = planner.plan_fft(fft_size, rustfft::FftDirection::Inverse);
        FftCorrelate64 {
            fft_size: fft_size,
            //planner: planner,
            forward: forward,
            inverse: inverse,
        }
    }

    pub fn correlate(
        &self, 
        a: &[Complex<f64>],
        b: &[Complex<f64>]) -> Vec<Complex<f64>>
    {
        let mut ap = vec!(
            Complex::<f64> {
                re: 0.0, im: 0.0 
            }; self.fft_size
        );
        
        let mut bp = vec!(
            Complex::<f64> {
                re: 0.0, im: 0.0
            }; self.fft_size
        );

        for (avv, av) in ap.iter_mut().zip(a.iter()) {
            *avv = *av;
        }

        for (bvv, bv) in bp.iter_mut().zip(b.iter()) {
            *bvv = *bv;
        }

        self.forward.process(&mut ap);
        self.forward.process(&mut bp);

        for (av, bv) in ap.iter_mut().zip(bp.iter()) {
            *av = *av * bv.conj();
        }

        self.inverse.process(&mut ap);
        
        let adj = (b.len() - 1) * 2;
        let valid = a.len() - b.len() + 1;

        let mut out: Vec<Complex<f64>> = Vec::with_capacity(
            valid
        );

        for x in 0..valid {
            out.push(
                ap[(adj + x) % ap.len()] / ap.len() as f64
            );
        }
        
        out
    }
}

pub fn linspace64(start: f64, end: f64, steps: u32) -> Vec<f64> {
    let mut out: Vec<f64> = vec!(0.0f64; steps as usize);
    let delta = end - start;
    
    let mut step = 0u32;
    for out_v in out.iter_mut() {
        let p = (step as f64) / (steps as f64 - 1.0);
        let cur = start + p * delta;
        *out_v = cur;
        step += 1;
    }

    out
}

pub fn linspace32(start: f32, end: f32, steps: u32) -> Vec<f32> {
    let mut out: Vec<f32> = vec!(0.0f32; steps as usize);
    let delta = end - start;
    
    let mut step = 0u32;
    for out_v in out.iter_mut() {
        let p = (step as f32) / (steps as f32 - 1.0);
        let cur = start + p * delta;
        *out_v = cur;
        step += 1;
    }

    out
}

pub fn convert_iqi16_to_iqf32(a: &[Complex<i16>], out: &mut [Complex<f32>]) {
    for (av, out_v) in a.iter().zip(out.iter_mut()) {
        (*out_v).re = av.re as f32;
        (*out_v).im = av.im as f32;
    }
}

pub fn convert_iqi16_to_iqf64(a: &[Complex<i16>], out: &mut [Complex<f64>]) {
    for (av, out_v) in a.iter().zip(out.iter_mut()) {
        (*out_v).re = av.re as f64;
        (*out_v).im = av.im as f64;
    }
}

pub fn div_iqf32_scalar_inplace(a: &mut [Complex<f32>], b: f32) {
    for v in a.iter_mut() {
        *v /= b;
    }
}