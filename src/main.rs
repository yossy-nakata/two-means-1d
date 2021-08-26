extern crate rand;
extern crate twomeans1d;
use plotters::prelude::*;
use rand::Rng;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use twomeans1d::*;
/// データファイルを読み込む
fn load_data(fname: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rslt = vec![];
    let mut fpath = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    fpath.push("data");
    fpath.push(fname);
    let fs = File::open(fpath)?;
    let reader = BufReader::new(fs);
    for line in reader.lines() {
        rslt.push(line?.parse()?);
    }
    Ok(rslt)
}
/// 散布図をプロット
fn plot(fname: &str, buf1: &[f64], buf2: &[f64]) -> Result<(), Box<dyn Error>> {
    let fname = format!("{}~.png", fname);
    let mut rng = rand::thread_rng();
    let root = BitMapBackend::new(&fname, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .build_cartesian_2d(50.0f64..200.0f64, -3.0f64..4.0f64)?;

    chart
        .configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    chart.draw_series(
        buf1.iter()
            .map(|x| Circle::new((*x, rng.gen()), 3, RED.filled())),
    )?;
    chart.draw_series(
        buf2.iter()
            .map(|x| Circle::new((*x, rng.gen()), 3, BLUE.filled())),
    )?;

    Ok(())
}
/// データの分割
fn split(data: &[f64], labels: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let mut buf1 = vec![];
    let mut buf2 = vec![];
    for i in 0..labels.len() {
        if labels[i] == 0 {
            buf1.push(data[i]);
        }
        if labels[i] == 1 {
            buf2.push(data[i]);
        }
    }
    (buf1, buf2)
}

/// メイン処理
fn main() -> Result<(), Box<dyn Error>> {
    let fnames = vec![
        "1_norm100_10x20-norm140_10x100.txt",
        "2_norm100_10x20-norm150_10x100.txt",
        "3_norm100_10x20-norm160_10x100.txt",
    ];

    for fname in fnames.iter() {
        println!();
        let data = load_data(fname)?;
        let (labels, cluster, k1bic, k2bic) = TwoMeans1D::analyse(&data);
        println!("k=1 bic : {}", &k1bic);
        println!("k=2 bic : {}", &k2bic);
        println!("the number of cluster : {}", &cluster.len());
        println!("cluster centers:{:?}", &cluster);
        let (v1, v2) = split(&data, &labels);
        let out = &fname[..9];
        plot(out, &v1, &v2)?;
    }

    Ok(())
}
