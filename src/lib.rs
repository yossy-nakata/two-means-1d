/// Two Means Clustering for 1d data
/// このプログラムは X-means クラスタリングの機能限定版です。
/// プライスヒストグラムの多峰性検知用に実験的に作られました。
///
/// References:
/// - Ishioka, T. (2005): An Expansion of X-means for Automatically Determining the Optimal Number of Clusters
///	  http://www.rd.dnc.ac.jp/%7Etunenori/doc/487-053.pdf
extern crate statrs;
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::PI;

pub struct TwoMeans1D {}

impl TwoMeans1D {
    /// 解析処理
    pub fn analyse(input: &[f64]) -> (Vec<usize>, Vec<f64>, f64, f64) {
        // 事前にソートする
        let mut temp = input.to_vec();
        temp.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // k-meansの初期値を生成
        let n = temp.len();
        let mid = 0.5 * (temp[0] + temp[n - 1]);
        let cluster1 = vec![mid];
        let cluster2 = vec![temp[0], temp[n - 1]];
        // k=1とk=2でクラスタリング
        let (l1, c1) = Self::fit(input, &cluster1);
        let (l2, c2) = Self::fit(input, &cluster2);

        // k=2 についてはクラスター別にデータを分離する
        let mut x1: Vec<f64> = vec![];
        let mut x2: Vec<f64> = vec![];
        for i in 0..n {
            if l2[i] == 0 {
                x1.push(input[i]);
            } else {
                x2.push(input[i]);
            }
        }
        // k=1とk=2のBICを求める。
        let prior_bic = Self::prior_bic(input, c1[0], 2.);
        let post_bic = Self::post_bic(&x1, c2[0], &x2, c2[1], 2.);
        // BICの小さかったクラスタリング結果を返す。
        if prior_bic <= post_bic {
            (l1, c1, prior_bic, post_bic)
        } else {
            (l2, c2, prior_bic, post_bic)
        }
    }
    /// k-means clustering
    pub fn fit(input: &[f64], cluster: &[f64]) -> (Vec<usize>, Vec<f64>) {
        let mut labels = vec![0usize; input.len()];
        let mut cluster = cluster.to_vec();
        cluster.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // 収束するまでループ（１００回まで）
        let mut cnt = 0usize;
        while cnt < 100 {
            cnt += 1;
            // クラスターの割当
            for n in 0..input.len() {
                let mut dist_min = f64::MAX;
                for (k, _) in cluster.iter().enumerate() {
                    let dist = (input[n] - cluster[k]).abs();
                    if dist_min > dist {
                        dist_min = dist;
                        labels[n] = k;
                    }
                }
            }
            // クラスターの更新
            let prev_cluster = cluster.clone();
            for (k, cl) in cluster.iter_mut().enumerate() {
                let mut total = 0f64;
                let mut num = 0usize;
                for n in 0..labels.len() {
                    if k == labels[n] {
                        total += input[n];
                        num += 1;
                    }
                }
                if num > 0 {
                    *cl = total / num as f64;
                }
            }
            // 更新結果に変化がない場合は脱出
            let mut is_stop = true;
            for k in 0..cluster.len() {
                if (cluster[k] - prev_cluster[k]).abs() > f64::EPSILON {
                    is_stop = false;
                    break;
                }
            }

            if is_stop {
                break;
            }
        }
        (labels, cluster)
    }
    /// Log Likelihood
    pub fn log_likelihood(var: f64, n: f64) -> f64 {
        -n / 2. * ((2. * PI).ln() + var.ln()) - (n - 1.) / 2.
    }
    /// k=1 のBICを計算
    pub fn prior_bic(x: &[f64], center: f64, q: f64) -> f64 {
        let var = Self::var(x, center);
        let n = x.len() as f64;
        let lnl = Self::log_likelihood(var, n);
        -2.0 * lnl + q * n.ln()
    }
    /// k=2 のBICを計算
    pub fn post_bic(x1: &[f64], center1: f64, x2: &[f64], center2: f64, q: f64) -> f64 {
        let var1 = Self::var(x1, center1);
        let var2 = Self::var(x2, center2);
        let n1 = x1.len() as f64;
        let n2 = x2.len() as f64;

        let lnl1 = Self::log_likelihood(var1, n1);
        let lnl2 = Self::log_likelihood(var2, n2);
        let mut beta = 0.0;
        if var1 > 0.0 && var2 > 0.0 {
            beta = (center1 - center2).abs() / (var1 + var2).sqrt();
        }
        let norm = Normal::new(0.0f64, 1.0f64).unwrap();
        let alpha = 0.5 / norm.cdf(beta);
        (-2. * lnl1 - 2. * lnl2) + 2. * q * (n1 + n2).ln() - 2. * (n1 + n2) * alpha.ln()
    }
    /// 分散を求める
    pub fn var(input: &[f64], center: f64) -> f64 {
        let mut rslt = 0f64;
        for x in input.iter() {
            rslt += (x - center).powi(2);
        }
        let n = input.len();
        rslt /= (n - 1) as f64;
        rslt
    }
}
