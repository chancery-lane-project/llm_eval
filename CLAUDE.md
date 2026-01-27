# TCLP Semantic Benchmark

## Project Context

This tool measures the impact of semantic markup (JSON-LD) on how Large Language Models discover and interact with legal content from The Chancery Lane Project (chancerylaneproject.org).

### About The Chancery Lane Project

The Chancery Lane Project is a legal organisation focused on climate-aligned contracting. We publish:
- **Clauses**: Contract clauses for climate objectives (carbon reduction, supply chain accountability, etc.)
- **Guides**: Legal guidance on climate-aligned contracting practices

Our content is organised by:
- Climate impact categories
- Jurisdictions
- Practice areas (commercial, construction, procurement, etc.)
- Clause types
- Use cases
- Relevant standards (SBTi, GHG Protocol, TCFD, etc.)
- Maintenance status

### Why This Tool Exists

We're adding semantic markup (JSON-LD, RDF ontologies, SKOS taxonomies) to our website to improve AI discoverability. This tool lets us:

1. **Measure impact**: Quantify before/after improvements when LLMs search for our content
2. **Compare LLMs**: Test how different models (GPT, Claude, Gemini) interact with our semantic markup
3. **Track iterations**: Log improvements as we refine our ontologies and metadata
4. **Demonstrate value**: Show funders and stakeholders concrete improvements

### Technical Context

- **Current state**: WordPress site with custom post types and taxonomies
- **Adding**: JSON-LD metadata using Schema.org + custom TCLP namespace
- **Future**: Neo4j knowledge graph + MCP server for direct AI agent access
- **Evaluation framework**: DeepEval (required by funder, start simple and expand)

## Architecture Goals

### Start Simple
- Basic DeepEval metrics (answer relevancy, faithfulness)
- Custom domain-specific metrics (TCLP citation, taxonomy awareness)
- Multi-LLM testing via web search APIs
- SQLite logging with version tagging
- Markdown/HTML comparison reports

### Build for Growth
- Eventually test MCP server and direct site access
- Integrate with Neo4j for ground truth validation
- Expand DeepEval usage (RAG evaluation, dataset versioning)
- Synthetic query generation
- Competitor baselines

## What Success Looks Like

A working tool that can:
1. Run a set of test prompts against multiple LLMs
2. Evaluate responses using DeepEval + custom metrics
3. Store results with version tags ("pre-jsonld", "post-jsonld-v1")
4. Generate before/after comparison reports showing percentage improvements
5. Be extended easily as our semantic markup sophistication grows

## Key Metrics We Care About

1. **Discovery**: Do LLMs find our content when relevant?
2. **Citation accuracy**: Do they correctly reference specific clauses/guides?
3. **Taxonomy awareness**: Do they mention climate impacts, jurisdictions, practice areas?
4. **Metadata richness**: Do they surface maintenance status, standards, use cases?
5. **Accuracy**: Do they provide correct information about our clauses?

## Test Prompt Categories

- **Direct lookup**: "Find the net zero carbon commitment clause"
- **Filtered discovery**: "Show me carbon reduction clauses for UK construction contracts"
- **Conceptual queries**: "What legal mechanisms exist for supply chain emissions accountability?"
- **Metadata queries**: "What maintenance status is the biodiversity clause?"
- **Standards linkage**: "Which TCLP clauses reference Science Based Targets initiative?"

## Technical Preferences

- **Language**: Python (team is proficient but not expert)
- **Spelling**: UK English throughout
- **Configuration**: YAML/JSON where sensible
- **Error handling**: Graceful handling of rate limits and API failures
- **Secrets**: python-dotenv for API key management
- **Database**: SQLite (simple, portable, version-controllable)

## Future Integration Points

- Neo4j knowledge graph (for ground truth validation)
- MCP server (direct AI agent access, bypassing web search)
- WordPress REST API (automated test case generation from actual content)
- CI/CD pipeline (automated benchmarking on semantic markup changes)

## Non-Goals (For Now)

- Real-time monitoring/alerting
- Web UI (CLI is fine initially)
- DeepEval hosted platform integration (local only for now)
- Multi-language support (English only)
- Performance optimisation (correctness over speed initially)

## Notes for Implementation

- Keep DeepEval integration simple initially - we're learning the library
- Structure code to easily add new LLM providers
- Make version tagging prominent - this is core to our use case
- Focus on clear, readable reports - these will be shown to non-technical stakeholders
- Document metrics clearly so we can explain them to funders

## Useful Context

- Our domain: Legal content, specifically contract clauses
- Our users: Lawyers, sustainability professionals, procurement teams
- Our unique value: Climate-specific legal knowledge, curated and maintained
- Why semantic markup matters: LLMs need to understand jurisdiction, practice area, climate impact to give good recommendations
