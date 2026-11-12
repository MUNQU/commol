//! Labels of the outputs of an aggregate model, the sums they name, and the
//! check that each name a model builds names one thing.

use std::collections::{BTreeMap, HashMap};

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
/// one thing here.
fn conflict(meanings: &[&Named]) -> bool {
    meanings
        .iter()
        .any(|named| named.owner != meanings[0].owner || !named.same_ids(meanings[0]))
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

/// `items`, two or more, joined by commas, and the last by `conjunction`.
///
/// A conflict has two meanings or more, and two ids or more to rename: two
/// owners, or an owner and a category, or, within one owner, the categories
/// of two lists of different ids, of which neither is a prefix of the other.
fn listed(items: &[String], conjunction: &str) -> String {
    let (last, rest) = items.split_last().expect("two or more items");
    format!("{} {conjunction} {last}", rest.join(", "))
}

/// Check that each name a model builds names one thing.
///
/// A name joins ids with `_`, and ids have no rule on their characters, so
/// two things can build one name: bin `A` in category `g1` of `group` and a
/// bin `A_g1` both build `A_g1`. The names are those of the outputs, the
/// compartments and the accumulators ([`OutputLabels::prefixes`]: an owner,
/// a prefix of its categories, one output), and those of `totals`: the bin
/// totals, the partial bin totals and the `total_<categories>`, each with what
/// it names. A name is refused when two owners build it, or when one owner
/// builds it from two lists of category ids. One owner may build it from one
/// list of category ids in two stratifications, as the subgroup total
/// `total_c1` when `group` and `type` both have `c1`.
///
/// # Errors
///
/// Returns a message with one sentence for each conflict that does not
/// follow from a conflict of shorter names (`follows_from_a_shorter`): what
/// the name names, and the bins, accumulators and categories to rename.
pub(crate) fn check_names<'a>(
    compartments: &'a [OutputLabels],
    accumulators: &'a [OutputLabels],
    totals: impl IntoIterator<Item = (String, Named<'a>)>,
) -> Result<(), String> {
    // Each name with what it names, and whether that is one output.
    let mut built: BTreeMap<String, Vec<(Named<'a>, bool)>> = BTreeMap::new();
    let mut add = |name: String, named: Named<'a>, whole: bool| {
        let meanings = built.entry(name).or_default();
        if !meanings.iter().any(|(other, _)| *other == named) {
            meanings.push((named, whole));
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
            add(
                name,
                Named { owner, categories },
                count == label.categories.len(),
            );
        }
    }
    for (name, named) in totals {
        add(name, named, false);
    }

    let name_of: HashMap<&Named, &str> = built
        .iter()
        .flat_map(|(name, meanings)| {
            meanings
                .iter()
                .map(move |(named, _)| (named, name.as_str()))
        })
        .collect();
    let sentences: Vec<String> = built
        .iter()
        .filter(|(_, meanings)| {
            let meanings: Vec<&Named> = meanings.iter().map(|(named, _)| named).collect();
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
}
