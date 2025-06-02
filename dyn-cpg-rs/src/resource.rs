use blake3;
use git2::Repository;
use once_cell::unsync::OnceCell;
use std::fs;
use std::path::PathBuf;

#[derive(Debug)]
pub enum Source {
    Worktree,
    Git(String),
    Empty,
}

#[derive(Debug)]
pub enum ResourceError {
    Io(std::io::Error),
    Utf8(std::string::FromUtf8Error),
    Git(git2::Error),
    NotFound(String),
}

impl From<std::io::Error> for ResourceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}
impl From<git2::Error> for ResourceError {
    fn from(e: git2::Error) -> Self {
        Self::Git(e)
    }
}
impl From<std::string::FromUtf8Error> for ResourceError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::Utf8(e)
    }
}

pub struct Resource {
    path: PathBuf,
    source: Source,
    repo: Option<Repository>,
    content_cache: OnceCell<Vec<u8>>,
    hash_cache: OnceCell<blake3::Hash>,
}

impl Resource {
    pub fn new<P: Into<PathBuf>>(path: P) -> Self {
        Self {
            path: path.into(),
            source: Source::Worktree,
            repo: None,
            content_cache: OnceCell::new(),
            hash_cache: OnceCell::new(),
        }
    }

    pub fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            source: Source::Empty,
            repo: None,
            content_cache: OnceCell::new(),
            hash_cache: OnceCell::new(),
        }
    }

    pub fn with_git<P: Into<PathBuf>>(
        mut self,
        rev: String,
        repo_root: P,
    ) -> Result<Self, ResourceError> {
        let repo = Repository::open(repo_root.into())?;
        self.repo = Some(repo);
        self.source = Source::Git(rev);
        Ok(self)
    }

    pub fn raw_path(&self) -> &PathBuf {
        &self.path
    }

    /// List files in the directory (if the resource is a directory)
    pub fn list(&self) -> Result<Vec<PathBuf>, ResourceError> {
        match &self.source {
            Source::Worktree => {
                if self.path.is_dir() {
                    let entries = fs::read_dir(&self.path)?
                        .filter_map(Result::ok)
                        .map(|e| e.path())
                        .collect();
                    Ok(entries)
                } else {
                    Err(ResourceError::NotFound(format!(
                        "Path is not a directory: {}",
                        self.path.display()
                    )))
                }
            }
            Source::Git(rev) => {
                let repo = self
                    .repo
                    .as_ref()
                    .ok_or_else(|| ResourceError::NotFound("Repository not initialized".into()))?;
                let object = repo.revparse_single(rev)?;
                let tree = object.peel_to_tree()?;

                // Traverse into self.path to get the sub-tree
                let sub_tree_entry = tree.get_path(&self.path)?;
                let sub_tree = repo.find_tree(sub_tree_entry.id())?;

                let entries = sub_tree
                    .iter()
                    .map(|e| {
                        e.name()
                            .map(PathBuf::from)
                            .ok_or_else(|| ResourceError::NotFound("Entry name is missing".into()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(entries)
            }

            Source::Empty => Ok(vec![]),
        }
    }

    /// This returns a result indicating whether the file definitely does or does not exist or a `ResourceError`
    /// Using `resource.exists().unwrap_or(false)` will return false if the resource errors
    pub fn exists(&self) -> Result<bool, ResourceError> {
        match &self.source {
            Source::Worktree => Ok(self.path.exists()),
            Source::Git(rev) => {
                let repo = self
                    .repo
                    .as_ref()
                    .ok_or_else(|| ResourceError::NotFound("Repository not initialized".into()))?;
                let object = repo.revparse_single(rev)?;
                let tree = object.peel_to_tree()?;
                let entry = tree.get_path(&self.path);
                Ok(entry.is_ok())
            }
            Source::Empty => Ok(true),
        }
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>, ResourceError> {
        self.content_cache
            .get_or_try_init(|| match &self.source {
                Source::Worktree => fs::read(&self.path).map_err(ResourceError::from),
                Source::Git(rev) => {
                    let repo = self.repo.as_ref().ok_or_else(|| {
                        ResourceError::NotFound("Repository not initialized".into())
                    })?;
                    let object = repo.revparse_single(rev)?;
                    let tree = object.peel_to_tree()?;
                    let entry = tree.get_path(&self.path)?;
                    let blob = repo.find_blob(entry.id())?;
                    Ok(blob.content().to_vec())
                }
                Source::Empty => Ok(vec![]),
            })
            .cloned()
    }

    pub fn read_string(&self) -> Result<String, ResourceError> {
        let bytes = self.read_bytes()?;
        Ok(String::from_utf8(bytes)?)
    }

    pub fn hash(&self) -> Result<blake3::Hash, ResourceError> {
        self.hash_cache
            .get_or_try_init(|| {
                let content = self.read_bytes()?;
                Ok(blake3::hash(&content))
            })
            .copied()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_git_directory() {
        let resource = Resource::new("dyn-cpg-rs")
            .with_git("HEAD".to_owned(), "./..")
            .expect("Repo must exist");

        assert!(resource.list().is_ok(), "Listing files should succeed");
    }

    #[test]
    fn test_git_file() {
        let r = Resource::new("dyn-cpg-rs/src/main.rs")
            .with_git("HEAD".to_owned(), "./..")
            .expect("Repo must exist");

        assert!(r.exists().unwrap_or(false), "File must exist");
        assert!(r.read_string().is_ok());

        println!("Hash: {:?}", r.hash().unwrap());
        println!("Content: {:?}", r.read_string().unwrap());
    }

    #[test]
    fn test() {
        let r1 = Resource::new("dyn-cpg-rs/src/main.rs")
            .with_git("HEAD".to_owned(), "./..")
            .expect("Repo must exist");

        let r2 = Resource::new("./src/main.rs");

        println!(
            "{:?}",
            r1.read_string().unwrap() == r2.read_string().unwrap()
        )
    }
}
