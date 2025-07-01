import pandas as pd

data_path = "data.csv"

chemistry_abstracts = [
    "This study investigates the catalytic properties of novel transition metal complexes for sustainable chemical reactions.",
    "We explore the synthesis and characterization of organic molecules with potential applications in pharmaceuticals.",
    "The effect of pH on the enzymatic activity in aqueous solutions was examined using spectroscopic methods.",
    "A new approach for the selective oxidation of alcohols catalyzed by metal-organic frameworks is presented.",
    "Thermodynamic properties of ionic liquids and their potential use as green solvents are analyzed.",
    "The photochemical behavior of photosensitizers in solar energy conversion devices was evaluated.",
    "Synthesis of nanoparticles with controlled size and morphology for drug delivery systems is discussed.",
    "We report the use of electrochemical methods to investigate corrosion resistance in metal alloys.",
    "The interaction between polymers and metal ions in aqueous media was studied to design better filtration membranes.",
    "Development of novel fluorescent probes for detecting heavy metals in environmental samples is described."
] * 20

data = pd.read_csv(data_path)

chem_df = pd.DataFrame({
    "abstract": chemistry_abstracts,
    "category": ["Chemistry"] * len(chemistry_abstracts)
})

updated_data = pd.concat([data, chem_df], ignore_index=True)
updated_data.to_csv("data_with_chemistry.csv", index=False)

print("Added Chemistry abstracts. New dataset shape:", updated_data.shape)
