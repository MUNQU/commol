//! Labels of the outputs of an aggregate model, the sums they name, and the
//! check that each name a model builds names one thing, and that no formula
//! reads a name that a shared category gives two meanings.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Separator of a qualified category id, as in `group:g1`.
pub const CATEGORY_SEPARATOR: char = ':';

/// Source and applied stratification categories of one simulation output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutputLabels {
    /// Bin or accumulator the output expands from.
    pub source: String,
    /// `(stratification id, category)` pairs applied to the output, in
    /// stratification declaration order.
    pub categories: Vec<(String, String)>,
}

impl OutputLabels {
    /// Names the output answers to as a prefix: its source, then its source
    /// followed by each leading category (`A`, `A_g1`, `A_g1_s2`). The last
    /// one is the output's own name.
    pub fn prefixes(&self) -> Vec<String> {
        let mut prefix = self.source.clone();
        let mut prefixes = vec![prefix.clone()];
        for (_, category) in &self.categories {
            prefix.push('_');
            prefix.push_str(category);
            prefixes.push(prefix.clone());
        }
        prefixes
    }

    /// Category applied to the output for a stratification, if any.
    pub fn category(&self, stratification: &str) -> Option<&str> {
        self.categories
            .iter()
            .find(|(id, _)| id == stratification)
            .map(|(_, category)| category.as_str())
    }
}

/// Every name that sums several outputs, with the indices of the outputs it
/// sums, sorted by name.
///
/// A name is an output's source followed by any number of its leading
/// categories (`A`, `A_g1`), short of its full name. It is built from the
/// labels rather than by splitting names, so a bin `A` never covers the
/// outputs of a bin `A_x`. A name is kept only when two or more outputs share
/// it. This covers bin totals over every category and partial totals over the
/// categories left unnamed.
pub fn output_sums(labels: &[OutputLabels]) -> Vec<(String, Vec<usize>)> {
    let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
    for (idx, label) in labels.iter().enumerate() {
        let mut prefixes = label.prefixes();
        prefixes.pop();
        for prefix in prefixes {
            map.entry(prefix).or_default().push(idx);
        }
    }
    let mut result: Vec<(String, Vec<usize>)> =
        map.into_iter().filter(|(_, v)| v.len() > 1).collect();
    result.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
    result
}

/// What builds a name: a bin, an accumulator, or the subgroup totals over
/// categories (`total_<categories>`), which take the compartments of every
/// bin.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Owner<'a> {
    /// A bin: its total, its prefixes, its compartments and its partial
    /// totals.
    Bin(&'a str),
    /// An accumulator: its total, its prefixes and its outputs.
    Accumulator(&'a str),
    /// The subgroup totals over categories.
    Categories,
}

/// What a name names: its owner and the categories the name takes, in
/// stratification declaration order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Named<'a> {
    /// The owner of the name.
    pub(crate) owner: Owner<'a>,
    /// The `(stratification id, category)` pairs the name takes.
    pub(crate) categories: Vec<&'a (String, String)>,
}

impl<'a> Named<'a> {
    /// The same owner without the category at `index`.
    fn without(&self, index: usize) -> Self {
        let mut categories = self.categories.clone();
        categories.remove(index);
        Self {
            owner: self.owner,
            categories,
        }
    }

    /// Whether `other` takes the same category ids, whatever their
    /// stratifications.
    fn same_ids(&self, other: &Self) -> bool {
        let ids = |named: &Self| -> Vec<&'a str> {
            named
                .categories
                .iter()
                .map(|(_, category)| category.as_str())
                .collect()
        };
        ids(self) == ids(other)
    }

    /// How a message names it; `whole` when it is one output.
    fn describe(&self, whole: bool) -> String {
        let categories: Vec<String> = self.categories.iter().map(|c| qualified(c)).collect();
        let (kind, id, one, several) = match self.owner {
            Owner::Bin(id) => ("bin", id, "compartment", "compartments"),
            Owner::Accumulator(id) => ("accumulator", id, "output", "outputs"),
            Owner::Categories => {
                return format!("the subgroup total of {}", categories.join(", "));
            }
        };
        if categories.is_empty() {
            return format!("the {kind} '{id}'");
        }
        let of = if whole { one } else { several };
        format!("the {of} of {kind} '{id}' in {}", categories.join(", "))
    }
}

/// A category as its qualified id, quoted: `'group:g1'`.
fn qualified((stratification, category): &(String, String)) -> String {
    format!("'{stratification}{CATEGORY_SEPARATOR}{category}'")
}

/// Whether `meanings` of one name are two things or more: two owners, or one
/// owner from two lists of category ids. One list of ids in two
/// stratifications, as `total_c1` when `group` and `type` both have `c1`, is
/// no conflict: a shared category gives the name two meanings ([`shared`]).
fn conflict(meanings: &[&Named]) -> bool {
    meanings
        .iter()
        .any(|named| named.owner != meanings[0].owner || !named.same_ids(meanings[0]))
}

/// Whether `meanings` of one name are two or more of one owner and one list
/// of category ids, in two stratifications: those that a category id two
/// stratifications share gives a name, as the totals of `group:c1` and of
/// `type:c1` give `total_c1`.
fn shared(meanings: &[&Named]) -> bool {
    meanings.len() > 1 && !conflict(meanings)
}

/// What each meaning of a name names, without whether it is one output.
fn named<'b, 'a>(meanings: &'b [(Named<'a>, bool)]) -> Vec<&'b Named<'a>> {
    meanings.iter().map(|(named, _)| named).collect()
}

/// Whether a conflict follows from a conflict of shorter names: without the
/// last category of each meaning, or without a category that every meaning
/// takes, the meanings are names of the model, of one name, that conflict.
fn follows_from_a_shorter(meanings: &[&Named], name_of: &HashMap<&Named, &str>) -> bool {
    let without_last: Option<Vec<Named>> = meanings
        .iter()
        .map(|named| {
            named
                .categories
                .len()
                .checked_sub(1)
                .map(|last| named.without(last))
        })
        .collect();
    let without_shared = meanings[0]
        .categories
        .iter()
        .filter(|category| {
            meanings
                .iter()
                .all(|named| named.categories.contains(category))
        })
        .map(|category| {
            meanings
                .iter()
                .map(|named| {
                    let index = named.categories.iter().position(|other| other == category);
                    named.without(index.expect("every meaning takes the category"))
                })
                .collect::<Vec<_>>()
        });
    without_last
        .into_iter()
        .chain(without_shared)
        .any(|shorter| {
            let names: Option<Vec<&str>> = shorter
                .iter()
                .map(|named| name_of.get(named).copied())
                .collect();
            let shorter: Vec<&Named> = shorter.iter().collect();
            names.is_some_and(|names| names.iter().all(|name| *name == names[0]))
                && conflict(&shorter)
        })
}

/// The sentence of a conflict: what `name` names, each of `meanings` with
/// whether it is one output, and the ids to rename. Those are the owners,
/// when they differ, and every category that not every meaning takes.
fn conflict_sentence(name: &str, meanings: &[(Named, bool)]) -> String {
    let described: Vec<String> = meanings
        .iter()
        .map(|(named, whole)| named.describe(*whole))
        .collect();
    let mut renames: Vec<String> = Vec::new();
    let mut push = |id: String| {
        if !renames.contains(&id) {
            renames.push(id);
        }
    };
    let first = meanings[0].0.owner;
    if meanings.iter().any(|(named, _)| named.owner != first) {
        for (named, _) in meanings {
            match named.owner {
                Owner::Bin(id) => push(format!("bin '{id}'")),
                Owner::Accumulator(id) => push(format!("accumulator '{id}'")),
                Owner::Categories => {}
            }
        }
    }
    meanings
        .iter()
        .flat_map(|(named, _)| named.categories.iter())
        .filter(|category| {
            !meanings
                .iter()
                .all(|(named, _)| named.categories.contains(category))
        })
        .for_each(|category| push(format!("category {}", qualified(category))));
    format!(
        "'{name}' names {}. Rename {}, so that each has a name of its own.",
        listed(&described, "and"),
        listed(&renames, "or")
    )
}

/// `items`, one or more, joined by commas, and the last by `conjunction`.
///
/// A conflict has two meanings or more, and two ids or more to rename: two
/// owners, or an owner and a category, or, within one owner, the categories
/// of two lists of different ids, of which neither is a prefix of the other.
/// The names of two meanings that one formula reads
/// ([`Names::check_reads`]) can be a list of one.
fn listed(items: &[String], conjunction: &str) -> String {
    match items.split_last().expect("one item or more") {
        (last, []) => last.clone(),
        (last, rest) => format!("{} {conjunction} {last}", rest.join(", ")),
    }
}

/// The qualified categories of a meaning, quoted: `'group:a'`, or
/// `'group:a', 'type:b'`.
fn categories_of(named: &Named) -> String {
    let categories: Vec<String> = named.categories.iter().map(|c| qualified(c)).collect();
    categories.join(", ")
}

/// A sum of names among `options`, the names of one meaning of each
/// compartment, that takes each compartment of `pending` once and none of
/// `taken`; `None` when there is none. Each name with its compartments.
fn cover<'o, 'n>(
    pending: &[usize],
    options: &'o BTreeMap<usize, Vec<(&'n str, BTreeSet<usize>)>>,
    taken: &BTreeSet<usize>,
) -> Option<Vec<(&'n str, &'o BTreeSet<usize>)>> {
    let Some(first) = pending.iter().find(|index| !taken.contains(index)) else {
        return Some(Vec::new());
    };
    options[first]
        .iter()
        .filter(|(_, set)| set.is_disjoint(taken))
        .find_map(|(name, set)| {
            let mut chosen = cover(pending, options, &(taken | set))?;
            chosen.push((name, set));
            Some(chosen)
        })
}

/// What each name of a model names: the names that the outputs of its
/// compartments and of its accumulators ([`OutputLabels::prefixes`]: an
/// owner, a prefix of its categories, one output) and its subgroup totals
/// build, each with what it names and whether that is one output.
pub(crate) struct Names<'a> {
    /// The compartments, whose labels say what each name takes.
    compartments: &'a [OutputLabels],
    /// Each name with its meanings.
    built: BTreeMap<String, Vec<(Named<'a>, bool)>>,
}

impl<'a> Names<'a> {
    /// The names that `compartments`, `accumulators` and `totals` build:
    /// the bin totals, the partial bin totals and the `total_<categories>`,
    /// each with what it names.
    pub(crate) fn new<'t>(
        compartments: &'a [OutputLabels],
        accumulators: &'a [OutputLabels],
        totals: impl IntoIterator<Item = (&'t str, &'t Named<'a>)>,
    ) -> Self
    where
        'a: 't,
    {
        let mut built: BTreeMap<String, Vec<(Named<'a>, bool)>> = BTreeMap::new();
        let mut add = |name: &str, named: Cow<'_, Named<'a>>, whole: bool| {
            let meanings = match built.get_mut(name) {
                Some(meanings) => meanings,
                None => built.entry(name.to_string()).or_default(),
            };
            if !meanings.iter().any(|(other, _)| *other == *named) {
                meanings.push((named.into_owned(), whole));
            }
        };
        let outputs = compartments
            .iter()
            .map(|label| (label, Owner::Bin(label.source.as_str())))
            .chain(
                accumulators
                    .iter()
                    .map(|label| (label, Owner::Accumulator(label.source.as_str()))),
            );
        for (label, owner) in outputs {
            for (count, name) in label.prefixes().into_iter().enumerate() {
                let categories = label.categories[..count].iter().collect();
                let named = Named { owner, categories };
                add(&name, Cow::Owned(named), count == label.categories.len());
            }
        }
        for (name, named) in totals {
            add(name, Cow::Borrowed(named), false);
        }
        Self {
            compartments,
            built,
        }
    }

    /// Refuse a name that two things build.
    ///
    /// A name joins ids with `_`, and ids have no rule on their characters,
    /// so two things can build one name: bin `A` in category `g1` of `group`
    /// and a bin `A_g1` both build `A_g1`. A name is refused when two owners
    /// build it, or when one owner builds it from two lists of category ids.
    /// One owner may build a name from one list of category ids in two
    /// stratifications, as the subgroup total `total_c1` when `group` and
    /// `type` both have `c1`: the name has a meaning in each stratification
    /// ([`shared`]), and no formula may read it ([`Self::check_reads`]).
    ///
    /// # Errors
    ///
    /// Returns a message with one sentence for each conflict that does not
    /// follow from a conflict of shorter names (`follows_from_a_shorter`):
    /// what the name names, and the bins, accumulators and categories to
    /// rename.
    fn check(&self) -> Result<(), String> {
        let name_of = self.name_of();
        let sentences: Vec<String> = self
            .built
            .iter()
            .filter(|(_, meanings)| {
                let meanings = named(meanings);
                conflict(&meanings) && !follows_from_a_shorter(&meanings, &name_of)
            })
            .map(|(name, meanings)| conflict_sentence(name, meanings))
            .collect();
        if sentences.is_empty() {
            return Ok(());
        }
        Err(format!(
            "Two things or more build one name, so a rate, an initial condition or a \
             result could read one for another: {}",
            sentences.join(" ")
        ))
    }

    /// The name of each meaning.
    fn name_of(&self) -> HashMap<&Named<'a>, &str> {
        self.built
            .iter()
            .flat_map(|(name, meanings)| {
                meanings
                    .iter()
                    .map(move |(named, _)| (named, name.as_str()))
            })
            .collect()
    }

    /// Whether a category id that two stratifications share gives `name`
    /// two meanings or more.
    fn is_shared(&self, name: &str) -> bool {
        self.built
            .get(name)
            .is_some_and(|meanings| shared(&named(meanings)))
    }

    /// The names to which a category id that two stratifications share gives
    /// two meanings or more. No formula reads them ([`Self::check_reads`]),
    /// and the engine sums none of them (`build_subgroup_total_mappings` in
    /// `crates/commol-model/src/builder.rs`). An accumulator gives no such
    /// name: each of its names is a prefix of its leading categories, of one
    /// list of categories, and a name that it shares with a total is a
    /// conflict, which [`Self::check`] refuses.
    pub(crate) fn shared_names(&self) -> HashSet<String> {
        self.built
            .keys()
            .filter(|name| self.is_shared(name))
            .cloned()
            .collect()
    }

    /// The compartments that `named` takes: those of its bin, or of every
    /// bin for a total over categories, with each of its categories.
    fn members(&self, named: &Named) -> BTreeSet<usize> {
        self.compartments
            .iter()
            .enumerate()
            .filter(|(_, label)| {
                (named.owner == Owner::Categories || named.owner == Owner::Bin(&label.source))
                    && named
                        .categories
                        .iter()
                        .all(|category| label.categories.contains(category))
            })
            .map(|(index, _)| index)
            .collect()
    }

    /// The names of one meaning whose sum takes each compartment of
    /// `meaning` once, in the order of their first compartment; `None` when
    /// no such sum is. A name takes compartment `c` when it is built from
    /// some of the categories of `c`: of its bin, or over categories when
    /// `meaning` is a total over categories.
    ///
    /// A compartment whose own name has one meaning reads as itself, so the
    /// search only chooses among names for the others ([`cover`]), and then
    /// takes the largest name left for each compartment.
    fn sum_of<'s>(
        &'s self,
        meaning: &Named<'a>,
        name_of: &HashMap<&'s Named<'a>, &'s str>,
    ) -> Option<Vec<&'s str>> {
        let wanted = self.members(meaning);
        let mut members: HashMap<&'s str, BTreeSet<usize>> = HashMap::new();
        let mut options: BTreeMap<usize, Vec<(&'s str, BTreeSet<usize>)>> = BTreeMap::new();
        let mut alone: Vec<usize> = Vec::new();
        for &index in &wanted {
            let label = &self.compartments[index];
            let bin = Owner::Bin(label.source.as_str());
            let owners: &[Owner] = if meaning.owner == Owner::Categories {
                &[bin, Owner::Categories]
            } else {
                &[bin]
            };
            let mut found: Vec<(&'s str, BTreeSet<usize>)> = Vec::new();
            let full_mask: u32 = (1 << label.categories.len()) - 1;
            for mask in 0..=full_mask {
                let categories: Vec<&(String, String)> = label
                    .categories
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (mask >> k) & 1 == 1)
                    .map(|(_, category)| category)
                    .collect();
                for &owner in owners {
                    let candidate = Named {
                        owner,
                        categories: categories.clone(),
                    };
                    let Some(&name) = name_of.get(&candidate) else {
                        continue;
                    };
                    let set = members
                        .entry(name)
                        .or_insert_with(|| self.members(&candidate));
                    if self.built[name].len() == 1 && set.is_subset(&wanted) {
                        found.push((name, set.clone()));
                    }
                }
            }
            found.sort_by(|(a, x), (b, y)| y.len().cmp(&x.len()).then(a.cmp(b)));
            let own = label.prefixes().pop().expect("the name of the compartment");
            if !found.iter().any(|(name, _)| *name == own) {
                alone.push(index);
            }
            options.insert(index, found);
        }
        let mut chosen: Vec<(&'s str, BTreeSet<usize>)> =
            cover(&alone, &options, &BTreeSet::new())?
                .into_iter()
                .map(|(name, set)| (name, set.clone()))
                .collect();
        let mut taken: BTreeSet<usize> = chosen.iter().flat_map(|(_, set)| set).copied().collect();
        for index in &wanted {
            if taken.contains(index) {
                continue;
            }
            let (name, set) = options[index]
                .iter()
                .find(|(_, set)| set.is_disjoint(&taken))
                .expect("the own name of the compartment takes it alone");
            taken.extend(set);
            chosen.push((name, set.clone()));
        }
        chosen.sort_by_key(|(_, set)| set.first().copied());
        Some(chosen.into_iter().map(|(name, _)| name).collect())
    }

    /// How to read each meaning of the shared name `name` with names of one
    /// meaning: as the sum of them that takes its compartments, or, when
    /// there is none, only after a rename.
    fn readings<'s>(&'s self, name: &str, name_of: &HashMap<&'s Named<'a>, &'s str>) -> String {
        let mut sums: Vec<String> = Vec::new();
        let mut renames = String::new();
        for (meaning, _) in &self.built[name] {
            let categories = categories_of(meaning);
            match self.sum_of(meaning, name_of) {
                Some(terms) => sums.push(format!("{categories} is '{}'", terms.join(" + "))),
                None if self.members(meaning).len() == 1 => renames.push_str(&format!(
                    " Only a rename reads {categories}, which has no other name."
                )),
                None => renames.push_str(&format!(
                    " Only a rename reads {categories}, which is no sum of names of one meaning."
                )),
            }
        }
        let sums = if sums.is_empty() {
            String::new()
        } else {
            format!(
                " To keep the ids, read a meaning as a sum of names of one meaning: {}.",
                sums.join("; ")
            )
        };
        format!("{sums}{renames}")
    }

    /// Check that no formula of `reads`, each a transition or a formula
    /// parameter with the variables its formulas read, reads a name that a
    /// shared category gives two meanings: a rate reads one value, and the
    /// name does not say which meaning it is.
    ///
    /// # Errors
    ///
    /// Returns a message that names each transition or formula parameter
    /// that reads such a name, with the names it reads, and for each name
    /// what it names, the categories to rename and how to read each meaning
    /// with names of one meaning, if it can be read so.
    pub(crate) fn check_reads(&self, reads: &[(String, Vec<String>)]) -> Result<(), String> {
        let mut read: BTreeSet<&str> = BTreeSet::new();
        let readers: Vec<String> = reads
            .iter()
            .filter_map(|(reader, variables)| {
                let names: BTreeSet<&str> = variables
                    .iter()
                    .map(String::as_str)
                    .filter(|name| self.is_shared(name))
                    .collect();
                read.extend(&names);
                let quoted: Vec<String> = names.iter().map(|name| format!("'{name}'")).collect();
                (!quoted.is_empty()).then(|| format!("{reader} reads {}", listed(&quoted, "and")))
            })
            .collect();
        if readers.is_empty() {
            return Ok(());
        }
        let name_of = self.name_of();
        let sentences: Vec<String> = read
            .iter()
            .map(|name| {
                let sentence = conflict_sentence(name, &self.built[*name]);
                format!("{sentence}{}", self.readings(name, &name_of))
            })
            .collect();
        Err(format!(
            "A rate or a formula parameter reads a name to which a category id that two \
             stratifications share gives two meanings or more: {}. {}",
            readers.join("; "),
            sentences.join(" ")
        ))
    }
}

/// Check that each name a model builds names one thing ([`Names::check`]),
/// and return what each names, for [`Names::check_reads`].
///
/// The names are those of the outputs, the compartments and the
/// accumulators ([`OutputLabels::prefixes`]: an owner, a prefix of its
/// categories, one output), and those of `totals`: the bin totals, the
/// partial bin totals and the `total_<categories>`, each with what it names.
///
/// # Errors
///
/// Returns the message of [`Names::check`].
pub(crate) fn check_names<'a, 't>(
    compartments: &'a [OutputLabels],
    accumulators: &'a [OutputLabels],
    totals: impl IntoIterator<Item = (&'t str, &'t Named<'a>)>,
) -> Result<Names<'a>, String>
where
    'a: 't,
{
    let names = Names::new(compartments, accumulators, totals);
    names.check()?;
    Ok(names)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(source: &str, categories: &[(&str, &str)]) -> OutputLabels {
        OutputLabels {
            source: source.to_string(),
            categories: categories
                .iter()
                .map(|(id, category)| (id.to_string(), category.to_string()))
                .collect(),
        }
    }

    #[test]
    fn sums_follow_labels_not_name_prefixes() {
        let labels = vec![
            label("A", &[("group", "g1"), ("sub", "s1")]),
            label("A", &[("group", "g1"), ("sub", "s2")]),
            label("A", &[("group", "g2"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s1")]),
            label("A_x", &[("group", "g1"), ("sub", "s2")]),
        ];

        let sums: std::collections::HashMap<String, Vec<usize>> =
            output_sums(&labels).into_iter().collect();

        assert_eq!(sums["A"], vec![0, 1, 2]);
        assert_eq!(sums["A_g1"], vec![0, 1]);
        assert_eq!(sums["A_x"], vec![3, 4]);
        assert_eq!(sums["A_x_g1"], vec![3, 4]);
        assert!(!sums.contains_key("A_g2"));
    }

    #[test]
    fn a_compartment_whose_own_name_has_two_meanings_reads_as_another_name_of_it() {
        // `A_a` is the one compartment of `A` and a partial total of `t:a`,
        // so the bin total `A` is the one name of one meaning that reads it.
        let labels = vec![
            label("A", &[("g", "a")]),
            label("B", &[("g", "b"), ("t", "a")]),
        ];
        let in_t = ("t".to_string(), "a".to_string());
        let totals = [(
            "A_a".to_string(),
            Named {
                owner: Owner::Bin("A"),
                categories: vec![&in_t],
            },
        )];
        let names = Names::new(
            &labels,
            &[],
            totals.iter().map(|(name, named)| (name.as_str(), named)),
        );
        let compartment = Named {
            owner: Owner::Bin("A"),
            categories: vec![&labels[0].categories[0]],
        };

        assert!(names.is_shared("A_a"));
        assert_eq!(
            names.sum_of(&compartment, &names.name_of()),
            Some(vec!["A"])
        );
    }

    #[test]
    fn a_name_that_no_sum_reads_in_any_meaning_is_read_only_after_a_rename() {
        // Two compartments of `A` are both `A_a`, and `A` sums them both.
        let labels = vec![label("A", &[("g", "a")]), label("A", &[("t", "a")])];
        let names = Names::new(&labels, &[], []);

        assert_eq!(
            names.readings("A_a", &names.name_of()),
            " Only a rename reads 'g:a', which has no other name. Only a rename reads 't:a', \
             which has no other name."
        );
    }
}
