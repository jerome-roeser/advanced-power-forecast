
run_main:
	python -m power.interface.main

streamlit:
	@streamlit run ui/app.py

reinstall_package:
	@pip uninstall -y power || :
	@pip install -e .
