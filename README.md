# Analysis of GitHub repository and project content.

The following repository contains an A.I workflow to generate summaries of GitHub repositorie's code and work over an X amount of time. 

The pipelines allow customizing what model conducts which summary, allowing the use of cheaper older models for low-level tasks, and newer costly models for high-level tasks (this is up to the user). 

You can change time (for activity) and models used in the [src/full.sh](src/full.sh) script. 

Please note that projects_seeds.csv is created within the pipeline, and if you want to test the pipeline with other or less projects, adjust[src/full.sh](src/full.sh) accoridngly.

Also note that the [src/full.sh](src/full.sh) pipeline removes all documents currently in reports and data folders. Plan accordingly before running.

## 1. First add the 2 needed tokens: one for GitHub and one for OpenAI API.

### - GitHub: 
#### - Go to [GitHub.com](https://github.com/) > Seitings > Developer settings > Personal Access Tokens > Tokens (classic) > Generate a token, select repo box and read:org box. You need to have nih-cfde oganization access.
### - OpenAI API: 
#### - Go to [OpenAI API](https://openai.com/api/pricing/), generate token to use 
### - Add tokens to the .env file (.env_example can serve as template)

## 2. Run the full pipeline: 

```bash
docker build -t cfde-pipeline --no-cache .

docker run --rm --env-file .env \
  -v "$PWD/data:/app/data" \
  -v "$PWD/reports:/app/reports" \
  -v "$PWD/reports_pdf:/app/reports_pdf" \
  cfde-pipeline
  ```

## Details
### 1) Clean outputs
- Removes files from data, reports, and reports_pdf. This is needed since data/summaries in these files are used (i.e., repo summaries are used for project summaries, etc.).

### 2) src/build_projects_seed.py
- This script grabs the JSON information from CFDE-Eval core private repository with repository and project information (i.e., what projects we care about). If you want to run pipeline with another project cohort, update project_seed.csv instead and don't run this.

### 3) /src/fetch_github_activity.py
- Uses GraphQL querying to fetch all github activity from all repostiroies in project_seed.csv. Fills /data/ folder. If you want to use another time, update that call: --days=365. If GraphQL fails for repos, there are retries implemented. Failure is still a possibility (netweork issues, etc.).

### 4) /src/normalize_activity.py and /src/rollup_projects.py
- These files put the needed information in a digestible format for LLM (parquet files and JSON structures)

### 5) LLM calls: /src/summarize_repos.py, /src/summarize_projects.py, /src/summarize_portfolio.py

#### /src/summarize_repos.py: 
- Generates per-repository executive-summary Markdown reports by loading cleaned GitHub activity tables and seed repos, shallow-cloning each repo to infer its goal from code, then prompting an OpenAI model (with retries) to synthesize “Summary and Goal” + “Recent Developments” sections and writing them to reports/, cleaning up clones afterward.

#### /src/summarize_projects.py:
- Aggregates repo-level “Summary and Goal” and “Recent Developments” sections from previously generated Markdown (with rollup JSON as fallback evidence) and uses an OpenAI model to synthesize a single per-project executive-summary Markdown report per project in reports/.

#### /src/summarize_portfolio.py:
Synthesizes a single portfolio-wide executive summary by reading the rollup _portfolio.json, pulling goal and “Recent Developments” text from project/repo Markdown reports (with metric-based fallback), then prompting an OpenAI model (with retries) to produce a two-section Markdown report written to reports/_portfolio_full.md.

### 6) src/make_pdf.py 
- Generates PDF files, saved in /reports_pdf/, from the markdown files generated before.


