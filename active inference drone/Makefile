.PHONY: test quick reproduce paper clean

test:
	python -m unittest discover -s tests -v

quick:
	python run_experiment.py --quick --workers 4 --output quick_results
	python run_collective_experiment.py --quick --workers 4 --output collective_quick_results

reproduce:
	python run_experiment.py --seeds 20 --seed-offset 1000 --workers 6 --output results
	python analyze_results.py
	python run_collective_experiment.py --seeds 20 --seed-offset 2000 --workers 6 --output collective_results

paper:
	mkdir -p paper/build_full
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build_full paper/v5_paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build_full paper/v5_paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=paper/build_full paper/v5_paper.tex

clean:
	find paper -maxdepth 2 -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' \) -delete

