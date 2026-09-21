"""
paper_generator.py
------------------
Fills the IEEEtran template with the project's REAL measured numbers and
writes paper/main.tex.

What changed, and why
~~~~~~~~~~~~~~~~~~~~~
This generator used to carry a hardcoded metrics dict as a fallback
(minADE5 0.81 m, 32.5 ms vs 115.2 ms baseline, 76% pruning, 98.5% accuracy
retention). Those numbers were never measured by anything in this repo. Worse,
the fallback was the ONLY code path that ever worked: it read `eval_*.json`
expecting keys like "minADE5" / "latency_ms" / "accuracy_retention", but
evaluate.py actually writes "minADE" / "avg_latency_ms_per_agent" and no
retention field at all, so a real results file raised KeyError and a missing
one produced a submission-ready paper full of invented results.

The generator now:
  * reads the schema evaluate.py actually writes,
  * raises instead of inventing anything when no results exist,
  * drops the "Ablation (No Fusion)" and "Ablation (No Audio)" rows, neither of
    which was ever run (audio is a dashboard TTS output, not a model input, so
    it cannot be ablated against a trajectory metric at all),
  * drops "accuracy retention", which has never been measured,
  * reports the latency delta with its real sign. Pruning currently makes
    inference SLOWER, because the decoder computes all K modes regardless of
    the mask; the paper says so rather than claiming a speed-up.
"""

import os
import json
import glob


class NoEvaluationResults(RuntimeError):
    """Raised instead of substituting invented numbers."""


class IEEEPaperGenerator:
    """Fills the IEEEtran double-column template with real evaluation numbers."""

    OURS_KEY = "Ours (HCP+MTR)"
    BASE_KEY = "No-HCP (MTR baseline)"

    def __init__(self, paper_dir, eval_dir):
        self.paper_dir = paper_dir
        self.eval_dir = eval_dir
        os.makedirs(paper_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def get_latest_eval_metrics(self):
        """Load the most recent evaluate.py output.

        Raises NoEvaluationResults rather than falling back to made-up values.
        """
        files = glob.glob(os.path.join(self.eval_dir, "eval_real_*.json"))
        if not files:
            raise NoEvaluationResults(
                f"No eval_real_*.json found in {self.eval_dir}. Run\n"
                f"  python hcp_project/eval/evaluate.py --checkpoint "
                f"hcp_project/outputs/mtr_checkpoint.pth --compare_hcp\n"
                f"first. This generator will not substitute placeholder metrics."
            )

        latest_file = max(files, key=os.path.getctime)
        with open(latest_file, "r") as f:
            raw = json.load(f)

        missing = [k for k in (self.OURS_KEY, self.BASE_KEY) if k not in raw]
        if missing:
            raise NoEvaluationResults(
                f"{latest_file} is missing {missing}. Re-run evaluate.py with "
                f"--compare_hcp so both the HCP and no-HCP configurations are measured."
            )

        return raw, latest_file

    # ------------------------------------------------------------------
    def generate_latex_document(self):
        metrics, source_file = self.get_latest_eval_metrics()
        ours = metrics[self.OURS_KEY]
        base = metrics[self.BASE_KEY]

        ours_ade = ours["minADE"]
        ours_fde = ours["minFDE"]
        ours_mr = ours["miss_rate_2m"]
        ours_lat = ours["avg_latency_ms_per_agent"]

        base_ade = base["minADE"]
        base_fde = base["minFDE"]
        base_mr = base["miss_rate_2m"]
        base_lat = base["avg_latency_ms_per_agent"]

        # Signed, honest. A positive number here would mean pruning is faster;
        # as the decoder stands it is not, so this comes out negative.
        latency_change = ((base_lat - ours_lat) / base_lat) * 100.0
        if latency_change >= 0:
            latency_sentence = (
                f"reducing per-agent inference latency by {latency_change:.1f}\\% "
                f"(from {base_lat:.1f}\\,ms to {ours_lat:.1f}\\,ms)"
            )
        else:
            latency_sentence = (
                f"at a per-agent latency \\emph{{cost}} of {abs(latency_change):.1f}\\% "
                f"(from {base_lat:.1f}\\,ms to {ours_lat:.1f}\\,ms), since the decoder "
                f"in its current form evaluates every mode regardless of the pruning mask"
            )

        # Whether pruning changed accuracy at all, stated as measured.
        if abs(ours_ade - base_ade) < 1e-6:
            accuracy_sentence = (
                "Pruning left minADE, minFDE and miss rate unchanged to machine "
                "precision: the mask is applied only to the mode confidence logits, "
                "while the reported minimum-over-modes metrics are by construction "
                "insensitive to confidence"
            )
        else:
            accuracy_sentence = (
                f"Pruning changed minADE from {base_ade:.2f}\\,m to {ours_ade:.2f}\\,m"
            )

        split_note = ours.get("scene_filter_used") or "the full training distribution"

        latex_template = r"""\documentclass[10pt,journal,compsoc]{IEEEtran}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning}

\begin{document}

\title{Hierarchical Combinatorial Pruning for Multimodal Motion Transformers:
A Negative Result on Candidate Pruning for Inference Cost}

\author{HCP-MTR Project Team}

\maketitle

\begin{abstract}
Trajectory prediction models typically score a fixed bank of candidate futures
for each agent, and a natural way to cut inference cost is to discard
implausible candidates before the expensive decoding step. We implement
Hierarchical Combinatorial Pruning (HCP), a three-stage cascade of kinematic
feasibility (KFF), spatial reachability (SRF), and social compatibility (SCF)
filters, in front of a Motion Transformer (MTR) core, and measure the
accuracy/cost trade-off on the real nuScenes dataset. We report a negative
result. __ACCURACY_SENTENCE__, and the cascade runs __LATENCY_SENTENCE__.
We identify the architectural reason: pruning that masks decoder outputs rather
than skipping decoder computation cannot reduce cost, and the
minimum-over-modes metrics standard in this literature cannot detect the
accuracy effect of such a mask. We state the conditions a pruning mechanism
must meet for the intended saving to be realisable.
\end{abstract}

\begin{IEEEkeywords}
Autonomous driving, motion forecasting, trajectory prediction, combinatorial
pruning, multimodal motion transformer, negative results.
\end{IEEEkeywords}

\section{Introduction}
\IEEEdropcaps{M}ODERN autonomous driving architectures rely on accurate,
real-time prediction of future trajectories for surrounding traffic
participants. Dense candidate models struggle to scale under strict latency
constraints, motivating cheap pre-decoding filters.

This paper reports what happened when we built one. Our contributions are:
\begin{itemize}
  \item \textbf{An implemented HCP cascade} (KFF, SRF, SCF) operating on
        candidates generated from agent history alone, with no ground-truth
        leakage.
  \item \textbf{A measured negative result:} the cascade does not reduce
        end-to-end inference cost, and we give the architectural reason.
  \item \textbf{A methodological observation:} minADE/minFDE/miss rate are
        insensitive to confidence-level pruning, so a pruning method evaluated
        only with these metrics can appear accuracy-neutral by construction
        rather than by merit.
\end{itemize}

\section{Related Work}
Recent trajectory forecasting models use Transformers for track encoding. The
Motion Transformer (MTR) uses learned intention points to query future paths.
Prior pruning techniques rely on distance heuristics; our cascade adds
kinematic and spatial constraints.

\section{Methodology}
\subsection{HCP Architecture}
The HCP module consists of three cascading filters:
\begin{enumerate}
  \item \textbf{KFF:} Eliminates paths violating curvature $\kappa \le 0.2$
        rad/m, jerk $j \le 5$ m/s$^3$, and lateral acceleration
        $a_{lat} \le 4$ m/s$^2$.
  \item \textbf{SRF:} Uses a KD-tree over map geometry to prune candidates
        that leave the drivable surface.
  \item \textbf{SCF:} Applies an agent-interaction collision check over
        candidate pairs.
\end{enumerate}

\begin{figure}[h]
\centering
\begin{tikzpicture}[node distance=1.5cm, auto]
\tikzstyle{block} = [rectangle, draw, fill=blue!10, text width=6.5em, text centered, rounded corners, minimum height=2em]
\tikzstyle{line} = [draw, -latex']
\node [block] (in) {Dense Candidates};
\node [block, below of=in] (kff) {Stage 1: KFF};
\node [block, below of=kff] (srf) {Stage 2: SRF};
\node [block, below of=srf] (scf) {Stage 3: SCF};
\node [block, below of=scf, fill=green!10] (out) {Sparse Set};
\path [line] (in) -- (kff);
\path [line] (kff) -- (srf);
\path [line] (srf) -- (scf);
\path [line] (scf) -- (out);
\end{tikzpicture}
\caption{Hierarchical Combinatorial Pruning (HCP) Cascade.}
\label{fig:hcp}
\end{figure}

\subsection{MTR Core and Cross-Attention Fusion}
Our backbone encodes agent track history and map polylines using RoPE
embeddings. The cross-attention layer adds a geometry bias
$B_{ij} = \text{MLP}(\text{RBF}(d_{ij}))$ to the attention matrix.

\section{Experiments}
We evaluate on the real nuScenes trainval split. Table~\ref{tab:results}
reports measured values over __N_AGENTS__ agent trajectories.

\begin{table}[h]
\centering
\caption{Measured results, HCP on vs.\ off}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
\textbf{Configuration} & \textbf{minADE} $\downarrow$ & \textbf{minFDE} $\downarrow$ & \textbf{MR@2m} $\downarrow$ & \textbf{ms/agent} $\downarrow$ \\
\midrule
No-HCP (MTR baseline) & __BASE_ADE__ & __BASE_FDE__ & __BASE_MR__ & __BASE_LAT__ \\
HCP + MTR & __OURS_ADE__ & __OURS_FDE__ & __OURS_MR__ & __OURS_LAT__ \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Threats to Validity}
These numbers carry substantial caveats and should not be read as competitive
benchmarks. Evaluation was performed on __SPLIT_NOTE__; the model was trained
without a held-out validation split, so these figures are not a measure of
generalisation. The absolute error magnitude indicates the predictor itself is
not yet converged, which limits what can be concluded about the pruning
cascade's effect on a well-trained model.

\section{Conclusion}
We implemented a three-stage combinatorial pruning cascade for a Motion
Transformer and measured its effect. The cascade does not reduce inference
cost, because masking decoder confidences does not skip decoder computation;
realising the intended saving requires a decoder that can evaluate a variable
subset of modes. We also observe that the minimum-over-modes metrics standard
in this literature cannot register the effect of confidence-level pruning,
which we believe is worth stating explicitly for future work in this direction.

\begin{thebibliography}{99}
\bibitem{nuscenes} nuScenes: A multimodal dataset for autonomous driving, IEEE CVPR, 2020.
\bibitem{mtr} Shi et al., Motion Transformer with Global Intention Localization, IEEE CVPR, 2022.
\bibitem{tnt} Zhao et al., TNT: Target-driven Trajectory Prediction, Conference on Robot Learning, 2020.
\end{thebibliography}

\end{document}
"""

        replacements = {
            "__ACCURACY_SENTENCE__": accuracy_sentence,
            "__LATENCY_SENTENCE__": latency_sentence,
            "__N_AGENTS__": f"{ours.get('num_agents_evaluated', 0):,}",
            "__SPLIT_NOTE__": split_note,
            "__BASE_ADE__": f"{base_ade:.2f}",
            "__BASE_FDE__": f"{base_fde:.2f}",
            "__BASE_MR__": f"{base_mr:.3f}",
            "__BASE_LAT__": f"{base_lat:.2f}",
            "__OURS_ADE__": f"{ours_ade:.2f}",
            "__OURS_FDE__": f"{ours_fde:.2f}",
            "__OURS_MR__": f"{ours_mr:.3f}",
            "__OURS_LAT__": f"{ours_lat:.2f}",
        }

        latex_out = latex_template
        for token, value in replacements.items():
            latex_out = latex_out.replace(token, value)

        leftover = [t for t in replacements if t in latex_out]
        assert not leftover, f"unfilled placeholders: {leftover}"

        main_tex_path = os.path.join(self.paper_dir, "main.tex")
        with open(main_tex_path, "w") as f:
            f.write(latex_out)
        print(f"Paper written to {main_tex_path} from measured results in {source_file}")
        return main_tex_path


if __name__ == "__main__":
    generator = IEEEPaperGenerator("hcp_project/paper", "hcp_project/outputs")
    try:
        generator.generate_latex_document()
    except NoEvaluationResults as exc:
        raise SystemExit(f"Refusing to generate a paper without measured results.\n\n{exc}")
