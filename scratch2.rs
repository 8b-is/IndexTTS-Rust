use std::f32::consts::PI;

fn main() {
    let num_frames = 100;
    let hop_length = 256;
    let frame_size = hop_length * 4;
    let output_length = (num_frames - 1) * hop_length + frame_size;
    let mut output = vec![0.0f32; output_length];
    let mut window_sum = vec![0.0f32; output_length];

    let mut window = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let t = i as f32 / (frame_size - 1) as f32;
        window[i] = 0.5 * (1.0 - (2.0 * PI * t).cos());
    }

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        
        let mut frame = vec![0.0f32; frame_size];
        for i in 0..frame_size {
            // Simulated sine wave
            frame[i] = (i as f32 * 0.1).sin();
        }

        for i in 0..frame_size {
            if start + i < output_length {
                output[start + i] += frame[i] * window[i];
                window_sum[start + i] += window[i] * window[i];
            }
        }
    }

    for i in 0..output_length {
        if window_sum[i] > 1e-8 {
            output[i] /= window_sum[i];
        }
    }

    // Find max in middle third
    let mid_start = output_length / 3;
    let mid_end = 2 * output_length / 3;
    let mut max_mid = 0.0;
    for i in mid_start..mid_end {
        if output[i].abs() > max_mid { max_mid = output[i].abs(); }
    }
    
    // Find max at edges
    let mut max_edge = 0.0;
    for i in 0..100 {
        if output[i].abs() > max_edge { max_edge = output[i].abs(); }
    }

    println!("Max in middle: {}", max_mid);
    println!("Max at edge: {}", max_edge);
}
