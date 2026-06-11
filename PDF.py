import os

# Define the HTML content for the Cover Letter with professional styling tailored for WeasyPrint.
# Colors: Navy blue accents (#0f2c59) for a commanding, scholarly, yet corporate tone appropriate for Cell Press.
# Page margins managed natively.

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    @page {
        size: A4;
        margin: 25mm 20mm;
        background-color: #ffffff;
        @bottom-right {
            content: "Page " counter(page);
            font-family: 'Times New Roman', Times, serif;
            font-size: 10pt;
            color: #666666;
        }
    }

    *, *::before, *::after {
        box-sizing: border-box;
    }

    body {
        margin: 0;
        padding: 0;
        font-family: 'Times New Roman', Times, serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #222222;
    }

    .sender-info {
        margin-bottom: 25px;
        font-size: 10.5pt;
        color: #333333;
        line-height: 1.4;
    }

    .date {
        margin-bottom: 25px;
        font-weight: bold;
    }

    .recipient-info {
        margin-bottom: 30px;
        line-height: 1.4;
    }

    .subject {
        margin-bottom: 25px;
        font-weight: bold;
        text-transform: uppercase;
        color: #0f2c59;
        border-bottom: 1px solid #0f2c59;
        padding-bottom: 5px;
    }

    .salutation {
        margin-bottom: 20px;
    }

    p {
        margin-top: 0;
        margin-bottom: 18px;
        text-align: justify;
    }

    .section-title {
        font-family: 'Times New Roman', Times, serif;
        font-size: 12pt;
        font-weight: bold;
        color: #0f2c59;
        margin-top: 25px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .signature-section {
        margin-top: 35px;
        page-break-inside: avoid;
    }

    .signature-line {
        margin-bottom: 20px;
    }
</style>
</head>
<body>

<div class="sender-info">
    <strong>Mingyue Wang</strong><br>
    School of Computer Science and Technology, Beijing Jiaotong University<br>
    No. 3 Shangyuancun, Haidian District, Beijing, 100044, China<br>
    Email: 23301048@bjtu.edu.cn
</div>

<div class="date">
    June 11, 2026
</div>

<div class="recipient-info">
    Editorial Board / Editor-in-Chief<br>
    <em>iScience</em>, Cell Press<br>
    50 Hampshire Street, 5th Floor<br>
    Cambridge, MA 02139, USA
</div>

<div class="subject">
    Subject: Submission of Original Research Manuscript for Publication in iScience
</div>

<div class="salutation">
    Dear Editors,
</div>

<p>
    On behalf of my co-authors, I am pleased to submit our original research manuscript titled <strong>"Cross-Platform Opinion Leader Identification on Social Media via Multimodal Decoupling and Contrastive Graph Attention Networks"</strong> for consideration for publication as an article in <em>iScience</em>.
</p>

<div class="section-title">What Was Previously Known (Background)</div>
<p>
    Accurately identifying opinion leaders on contemporary social media platforms is paramount for algorithmic public opinion governance, misinformation mitigation, and information diffusion analysis. Traditional computational paradigms heavily rely on structural network centrality indicators (e.g., PageRank, degree centrality) or shallow, fixed-weight linear feature fusion. However, these methods suffer from two critical bottlenecks: first, they fail to reconcile the intrinsic representation conflicts arising from heterogeneous multimodal features (e.g., linking network topology with fine-grained semantics), often leading to circular reasoning; second, they ignore the thematic heterogeneity of public opinion fields, operating under the flawed assumption that opinion leader formation follows a uniform structural pattern across all social contexts.
</p>

<div class="section-title">The Conceptual Advance Provided by Our Work</div>
<p>
    To transcend these limitations, this study presents a progressive, cross-disciplinary research framework that bridges advanced graph deep learning with behavioral communication science. Technically, we develop the Multimodal Decoupling and Contrastive Enhanced Graph Attention Network (MDCE-GAT). The model explicitly decouples semantic embeddings, network topology, and Large Language Model (LLM)-driven content quality scores into orthogonal feature spaces, utilizing self-supervised contrastive learning regularizers to mitigate structural noise and eliminate representation conflicts.
</p>
<p>
    Conceptually and empirically, our work provides a profound mechanistic advance by revealing that social media ecosystems exhibit bifurcated dynamics for opinion leader generation. Through systemic validation on real-world multi-platform datasets (Bilibili and Reddit), we empirically demonstrate that in emotionally resonant, fandom-driven topics, user influence is heavily dominated by network interaction topology and initial diffusion density, showing a stark head-concentration effect. Conversely, in value-driven, controversial public debates (e.g., online bullying), influence is fundamentally dictated by fine-grained content quality, logical rationality, and argumentation professionalism, while topological link density shows diminished predictive weight. MDCE-GAT successfully captures these distinct underlying dynamics adaptively without requiring hyperparameter reconfiguration.
</p>

<div class="section-title">Significance of the Work and Its Potential Impact</div>
<p>
    We believe our findings are highly aligned with the interdisciplinary scope of <em>iScience</em>, specifically appealing to researchers in data science, computational social science, and artificial intelligence. By introducing an architecture that dynamically calibrates representation weights based on the latent geometric properties of specific public opinion domains, this study moves beyond rigid algorithmic benchmarking to provide empirical substance to public opinion field stratification. The potential impact of this work is twofold: it provides a highly versatile, generalizable technical blueprint for cross-platform information cascade analysis, and offers platform regulators a precise, scenario-adaptive tool for targeted public opinion alignment and societal risk intervention.
</p>

<p>
    We confirm that this manuscript is original, has not been published elsewhere, and is not currently under consideration by any other journal. All authors have read and approved the final version of the manuscript for submission, and we declare no competing interests.
</p>

<p>
    Thank you very much for your time and consideration of our work. We look forward to hearing from you.
</p>

<div class="signature-section">
    <div class="signature-line">Sincerely,</div>
    <strong>Mingyue Wang</strong><br>
    Corresponding Author<br>
    School of Computer Science and Technology, Beijing Jiaotong University<br>
    Email: 23301048@bjtu.edu.cn
</div>

</body>
</html>
"""

# Save HTML file
html_path = 'iScience_Cover_Letter.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

# Convert to PDF using WeasyPrint
from weasyprint import HTML

pdf_path = 'iScience_Cover_Letter.pdf'
HTML(html_path).write_pdf(pdf_path)

print(f"PDF generated successfully at: {pdf_path}")