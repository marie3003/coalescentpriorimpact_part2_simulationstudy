import os, sys, io
import numpy as np
from Bio import Phylo, AlignIO, SeqIO  # SeqIO is used by MakeBEASTXML.py via `from beastutils import *`


def makeXMLFile(pars, template, outputfile="", outputpath=""):

	if (outputpath == ""):
		outputpath = pars['outputpath']

	if (not os.path.exists(outputpath)):
		os.makedirs(outputpath)

	if (outputfile == ""):
		outputfile = pars["name"]

	sys.stdout.write(outputfile+"...\n")

	formatpars = dict()
	for par in pars:
		formatpars['$'+par] = pars[par]
	output = template.format(**formatpars)

	outfile = open(outputpath+"/"+outputfile+".xml", 'w')
	outfile.write(output)
	outfile.close()


def pars_alignment(aln_path, pars):
    """
    Read a FASTA alignment and add it to pars["taxa"] in BEAST-compatible XML format.
    """
    alignment = AlignIO.read(aln_path, "fasta")
    output_align = io.StringIO()

    for record in alignment:
        output_align.write(f'\n\t\t<sequence><taxon idref="{record.id}"/>{str(record.seq)}</sequence>')

    pars["taxa"] = output_align.getvalue()

def pars_dates(tree, pars):
    """
    Add dates to the tree in the BEAST XML parameters.
    """
    output_dates = io.StringIO()

    tree_ = Phylo.read(io.StringIO(tree), "newick")
    leaves = tree_.get_terminals()

    # Get homochronous setting from pars (default to False if not present)
    is_homochronous = pars.get("homochronous", False)

    if is_homochronous:
        for leaf in leaves:
            output_dates.write(f'\n\t\t<taxon id = "{leaf.name}"> <date value="0.0" direction="backwards" units="years"/> </taxon>')
    else:
		# height is difference from root to tips, written at each tip
        heights = np.array([tree_.distance(leaf) for leaf in leaves])
        treetimes = np.max(heights) - heights
        for i, leaf in enumerate(leaves):
            output_dates.write(f'\n\t\t<taxon id = "{leaf.name}"> <date value="{treetimes[i]:.4f}" direction="backwards" units="years"/> </taxon>')

    pars["dates"] = output_dates.getvalue()
