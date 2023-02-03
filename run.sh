
rm -r configs/*
python scripts/generate_configs.py

for file in configs/*; do
  sh scripts/main.sh "$file"
done
