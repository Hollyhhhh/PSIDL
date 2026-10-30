"""PSIDL NONE codon simulation."""

import sys
import numpy
import scipy.stats
import random
from ete3 import Tree
from CodonZ import std_table
from CodonZ import codonListY as codon_list
import re
import os
from helper import initialize_run, choose_mode, fitness_path, record_branch, write_audit
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

save_dir = os.environ.get('PSIDL_SIM_OUTPUT', os.path.join(os.path.dirname(__file__), '..', 'outputs', os.path.basename(__file__).removesuffix('.py')))
tree_dir = save_dir +  '/' + 'newick' + '/'
try:
    os.makedirs(tree_dir, exist_ok=True)
except Exception:
    print(Exception)

codon_list = [x for x in codon_list if not x in ('TAA', 'TAG', 'TGA')]

number = sys.argv[1]

number = int(number)
initialize_run(number)
low_count = int(os.environ.get("PSIDL_SIM_CODONS", "1000"))
high_count = low_count
low_species = 24
high_species = 24
foreground_species_number = 8

def main():

    codon_count = numpy.random.randint(low_count, high=high_count+1)
    mode = choose_mode()

    species = numpy.random.randint(low_species, high=high_species+1)

    omega0 = numpy.random.beta(2, 5)
    # Background dN/dS follows Beta(2, 5).
    omega1 = 1 + numpy.random.gamma(2, 2.5)

    proportion = numpy.random.uniform(0.05, 0.95)
    branchlen_scaler = numpy.random.uniform(1, 5)

    tree = get_random_tree(species, branchlen_scaler, omega0, omega1, proportion)

    if mode == 'external':

        foreground_species = random.sample(range(0, species), int(foreground_species_number))
        foreground_list = numpy.zeros((high_species))
        for j in foreground_species:
            foreground_list[j] = 1
        foreground_nodes = []

    else:
        L = int(foreground_species_number)        # Target descendant-leaf count.
        k_min, k_max = 2, 7                      # Select two to seven internal branches.
        max_retry = 300

        internal_all = [n for n in tree.traverse()
                        if not n.is_leaf() and not n.is_root()]

        for _ in range(max_retry):
            random.shuffle(internal_all)
            foreground_nodes = []
            union_leaves = set()

            for node in internal_all:
                foreground_nodes.append(node)
                union_leaves |= set(node.iter_leaves())

                if len(foreground_nodes) > k_max:
                    break

                # Stop when the target descendant union is reached.
                if len(union_leaves) == L:
                    break

            # Require the target union and the selected-branch bounds.
            if len(union_leaves) == L and k_min <= len(foreground_nodes) <= k_max:
                break
        else:
            # Fallback: one internal branch with the target descendant count.
            foreground_nodes, union_leaves = None, None
            for n in internal_all:
                if len(set(n.iter_leaves())) == L:
                    foreground_nodes = [n]
                    union_leaves = set(n.iter_leaves())
                    break
            if foreground_nodes is None:
                raise RuntimeError("No internal foreground covering eight leaves was found")

        for idx, n in enumerate(foreground_nodes):
            if not n.name:
                n.name = f"internal{idx}"

        foreground_species = [
            int(re.findall(r'\d+', l.name)[0]) for l in union_leaves
        ]
        foreground_list = numpy.zeros((high_species))
        for j in foreground_species:
            foreground_list[j] = 1

    pi = equilibrium_generator(number)
    theta = theta_generator(number)
    evolve_matrix = get_q_matrix(pi, 1, codon_list, omega1, theta)
    middle_matrix = get_q_matrix(pi, 1, codon_list, omega0, theta)

    cdfreqs = make_cdfreqs(codon_list, pi)
    anc_freq = cdfreqs / numpy.sum(cdfreqs)
    ratio = numpy.floor(codon_count*numpy.array([[1-proportion, proportion]]))
    sites = [int(i) for i in range(codon_count)]
    random.shuffle(sites)
    neutral = sites[:int(ratio[0, 0])]
    selection = sites[int(ratio[0, 0]):]

    count = 0
    branch_decisions = []
    for node in tree.traverse(strategy='preorder'):
        # Propagate sequences from root to leaves.
        if node.is_root():
                # Draw ancestral codon states.
            node.dist = numpy.random.gamma(1, 0.2)*branchlen_scaler
            anc_seq = get_seq_by_multinomial(anc_freq, codon_count)
            node.add_feature("sequence", anc_seq)
        else:
            count += 1

            is_fg = False
            if mode == 'external':
                is_fg = node.is_leaf() and node.name in {
                    f"sequence{index}" for index in foreground_species
                }
            else:
                is_fg = (node in foreground_nodes)

            record_branch(branch_decisions, node, is_fg)

            if is_fg:

                node.dist = numpy.random.gamma(1, 0.2)*branchlen_scaler
                evol_pmat = get_evolved_pmat(scipy.linalg.expm(middle_matrix*node.dist), 1)
                evol_pmat_1 = get_evolved_pmat(scipy.linalg.expm(evolve_matrix*node.dist), 1)
                evol_pmat = numpy.vstack([evol_pmat, evol_pmat_1]).reshape(2, 61, 61)
                parent_seq = node.up.sequence
                parent_codon_pos = get_codon_positions(parent_seq)
                seq = get_sim_seq_branch(parent_seq, evol_pmat, parent_codon_pos, neutral, selection)
                node.add_feature("sequence", seq)
            else:

                node.dist = numpy.random.gamma(1, 0.2)*branchlen_scaler
                evol_pmat = get_evolved_pmat(scipy.linalg.expm(middle_matrix*node.dist), 1)
                parent_seq = node.up.sequence
                parent_codon_pos = get_codon_positions(parent_seq)
                node.add_feature("sequence", get_sim_seq(parent_seq, evol_pmat, parent_codon_pos))

    # Convert codon states to nucleotide sequences.

    for node in tree.traverse(strategy='preorder'):
        node.add_feature("sequence", ''.join([codon_list[x] for x in node.sequence]))
    tree.write(outfile=f'{tree_dir}tree{number}_none.rtree', format=1)

    tree.write(outfile=f'{tree_dir}seq{number}_none.newick', features=["sequence"], format=1)

    with open(f'{tree_dir}seq{number}_none.sites', 'w') as fr:
        fr.write(f'{selection}\n')

    branch_path = f'{tree_dir}seq{number}_none.branch'
    with open(branch_path, 'w') as fr:
        if mode == 'external':
            fr.write('LEAVES ' + ' '.join([f'sequence{idx}' for idx in foreground_species]) + '\n')
        else:
            fr.write('INTERNAL ' + ' '.join([n.name for n in foreground_nodes]) + '\n')
            internal_leaves = sorted({l.name for n in foreground_nodes for l in n.iter_leaves()})
            fr.write('LEAVES ' + ' '.join(internal_leaves) + '\n')

    write_audit(
        save_dir, number, "none", mode, codon_count, tree, branch_decisions,
        foreground_species, foreground_nodes,
        {
            "omega0": float(omega0),
            "omega1": float(omega1),
            "branch_length_scale": float(branchlen_scaler),
            "selected_site_fraction": float(proportion),
            "nucleotide_frequencies": pi.tolist(),
            "exchangeabilities": theta.tolist(),
        },
        fitness=None,
    )

def equilibrium_generator(sample):
    numpy.random.seed(sample)
    position1 = numpy.random.dirichlet(numpy.ones(4),size=1)
    position2 = numpy.random.dirichlet(numpy.ones(4),size=1)
    position3 = numpy.random.dirichlet(numpy.ones(4),size=1)
    equilibrium = numpy.concatenate([position1, position2, position3], axis=0)
    return equilibrium
def theta_generator(sample):
    numpy.random.seed(sample)
    theta_value = numpy.random.gamma(2, 0.5, 5)
    return theta_value

def get_sim_seq(anc_seq, evol_pmat, codon_pos):
    child_seq = anc_seq.copy()
    # Draw descendant codon states from the transition probabilities.
    for i in range(len(codon_list)):
        n = len(codon_pos[i])

        codon_child_state = get_seq_by_multinomial(evol_pmat[i], n)

        child_seq[codon_pos[i]] = codon_child_state

    return child_seq

def get_sim_seq_branch(anc_seq, evol_pmat, codon_pos, neutral, selection):
    child_seq = anc_seq.copy()

    for i in range(len(codon_list)):
        for j in range(2):
            if j == 0:
                local_sites = neutral
            else:
                local_sites = selection

            if numpy.any(numpy.intersect1d(codon_pos[i], local_sites)):
                local_child = numpy.intersect1d(codon_pos[i], local_sites)

            else:
                continue
            local_pmat = evol_pmat[j, :, :]
            n = len(local_child)
            if numpy.any(local_child):
                codon_child_state = get_seq_by_multinomial(local_pmat[i], n)

                child_seq[local_child] = codon_child_state

    return child_seq

def get_seq_by_multinomial(codon_freq, site_count):

    codon_freq = codon_freq / numpy.sum(codon_freq)
    custm = scipy.stats.rv_discrete(values=(range(len(codon_list)), codon_freq))

    return custm.rvs(size=site_count)
# Convert transition probabilities to cumulative sampling probabilities.

def get_q_matrix(pi, alpha, codon_list, omega , theta):
    ncd = len(codon_list)
    qmat = numpy.ones((ncd, ncd))
    for i in range(ncd):
        for j in range(ncd):
            diffL = [
                x for x in range(3) if codon_list[i][x] != codon_list[j][x]
            ]
            if len(diffL) > 1:
                qmat[i][j] = 0
            elif len(diffL) == 1:
                pos = diffL[0]
                pi_local = pi_determine(pos, codon_list[j], pi)
                theta_local = theta_determine(pos, codon_list[i], codon_list[j], theta)
                if std_table[codon_list[i]] == std_table[codon_list[j]]:
                    qmat[i][j] = alpha
                else:
                    qmat[i][j] = omega * alpha
                qmat[i][j] *= float(pi_local)
                qmat[i][j] *= float(theta_local)
    for i in range(ncd):
        qmat[i][i] = 0
        qmat[i][i] = -qmat[i, :].sum()
    cdfreqs = make_cdfreqs(codon_list, pi)
    cdfreqs = cdfreqs / numpy.sum(cdfreqs)
    scale_f = numpy.sum(cdfreqs * numpy.diag(qmat))
    qmat = qmat / abs(scale_f)
    return qmat
def theta_determine(pos, codon1, codon2, theta):
    if ''.join(codon1[pos]+codon2[pos]) == 'AC':
        theta_local = theta[0]
    elif ''.join(codon1[pos]+codon2[pos]) == 'CA':
        theta_local = theta[0]
    elif ''.join(codon1[pos]+codon2[pos]) == 'GT':
        theta_local = theta[1]
    elif ''.join(codon1[pos]+codon2[pos]) == 'TG':
        theta_local = theta[1]
    elif ''.join(codon1[pos]+codon2[pos]) == 'AT':
        theta_local = theta[2]
    elif ''.join(codon1[pos]+codon2[pos]) == 'TA':
        theta_local = theta[2]
    elif ''.join(codon1[pos]+codon2[pos]) == 'CG':
        theta_local = theta[3]
    elif ''.join(codon1[pos]+codon2[pos]) == 'GC':
        theta_local = theta[3]
    elif ''.join(codon1[pos]+codon2[pos]) == 'CT':
        theta_local = theta[4]
    elif ''.join(codon1[pos]+codon2[pos]) == 'TC':
        theta_local = theta[4]
    else :
        theta_local = 1 # The A-G exchangeability is fixed to one.
    return float(theta_local)
def make_cdfreqs(codon_list, pi):
    cdfreqs = []
    for i in range(len(codon_list)):
        freq1 = float(pi_determine(0, codon_list[i], pi))

        freq2 = float(pi_determine(1, codon_list[i], pi))

        freq3 = float(pi_determine(2, codon_list[i], pi))

        cdfreqs.append(freq1*freq2*freq3)

    return numpy.array(cdfreqs)
def pi_determine(pos, codon, pi):
    if codon[pos] == 'A':
        pi_local = pi[pos, 0]
    elif codon[pos] == 'C':
        pi_local = pi[pos, 1]
    elif codon[pos] == 'G':
        pi_local = pi[pos, 2]
    else:
        pi_local = pi[pos, 3]
    return float(pi_local)

def get_evolved_pmat(pmat,t):
    folded_p = pmat
    numpy.fill_diagonal(folded_p, 0)
    numpy.fill_diagonal(folded_p, 1-folded_p.sum(axis=1))
    return numpy.linalg.matrix_power(folded_p, t)
# Group alignment positions by their ancestral codon state.
def get_codon_positions(arr):
    codon_pos = []
    for codon_index in range(len(codon_list)):
        codon_pos.append([index for index, x in enumerate(arr) if x == codon_index])

    return codon_pos
def fileTurn(filedir):
    with open(filedir) as f:
        data = f.readlines()
    data = ''.join(data)
    data = re.findall('.+', data)
    for i in range(len(data)):
        data[i] = str(data[i]).split()
    return data
# Generate a bifurcating tree with nlv leaves.
def get_random_tree(nlv, branchlen_scaler, omega0, omega1, proportion):

    t = Tree()

    name_list = []
    for i in range(nlv):
        name_list.append('sequence'+str(i))

    t.populate(nlv, names_library=name_list, random_branches=True, branch_range=(1, 1))

    idx = 0
    for node in t.traverse():
        if not node.is_leaf() and not node.name:
            node.name = f'internal{idx}'
            idx += 1
    return t
def formated(data):
    data = ''.join(data)
    data = re.findall('.+',data)
    data = ''.join(data)
    data = data + ';'
    return data

if __name__ == '__main__':
    main()
