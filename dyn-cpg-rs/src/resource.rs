use blake3;
use git2::Repository;
use once_cell::sync::OnceCell;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug)]
pub enum Source {
    Worktree,
    Git(String),
    Empty,
}

#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("Git error: {0}")]
    Git(#[from] git2::Error),

    #[error("Resource not found: {0}")]
    NotFound(String),
}

pub struct Resource {
    path: PathBuf,
    source: Source,
    repo: Option<Repository>,
    content_cache: OnceCell<Vec<u8>>,
    hash_cache: OnceCell<blake3::Hash>,
}

impl Resource {
    /// The path will always be interpreted as relative to the current working directory unless it is absolute.
    /// Git paths will be truncated to be relative to the repository root internally, and must not be specified as relative to the root unless the repository root is the current working directory.
    pub fn new<P: Into<PathBuf>>(path: P) -> Result<Self, ResourceError> {
        Ok(Self {
            path: fs::canonicalize(&path.into())?,
            source: Source::Worktree,
            repo: None,
            content_cache: OnceCell::new(),
            hash_cache: OnceCell::new(),
        })
    }

    /// Creates an empty resource, which is useful for representing a non-existent or placeholder resource.
    pub fn empty() -> Self {
        Self {
            path: PathBuf::new(),
            source: Source::Empty,
            repo: None,
            content_cache: OnceCell::new(),
            hash_cache: OnceCell::new(),
        }
    }

    /// Sets the resource to be a Git resource with the specified revision and repository root.
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

    /// Returns the canonicalised path of the resource.
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
                let path = self
                    .path
                    .strip_prefix(repo.workdir().ok_or_else(|| {
                        ResourceError::NotFound("Repository workdir not found".into())
                    })?)
                    .map_err(|_| ResourceError::NotFound("Invalid path for repository".into()))?;
                let sub_tree_entry = tree.get_path(path)?;
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

    /// This returns a result indicating whether the file definitely does or does not exist or a ResourceError
    /// Using resource.exists().unwrap_or(false) will return false if the resource errors
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
                let path = self
                    .path
                    .strip_prefix(repo.workdir().ok_or_else(|| {
                        ResourceError::NotFound("Repository workdir not found".into())
                    })?)
                    .map_err(|_| ResourceError::NotFound("Invalid path for repository".into()))?;
                let entry = tree.get_path(path);
                Ok(entry.is_ok())
            }
            Source::Empty => Ok(true),
        }
    }

    /// Reads the content of the resource as bytes.
    /// This is cached, so subsequent calls will not re-read the content.
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
                    let path = self
                        .path
                        .strip_prefix(repo.workdir().ok_or_else(|| {
                            ResourceError::NotFound("Repository workdir not found".into())
                        })?)
                        .map_err(|_| {
                            ResourceError::NotFound("Invalid path for repository".into())
                        })?;
                    let entry = tree.get_path(path)?;
                    let blob = repo.find_blob(entry.id())?;
                    Ok(blob.content().to_vec())
                }
                Source::Empty => Ok(vec![]),
            })
            .cloned()
    }

    /// Reads the content of the resource as a UTF-8 string.
    /// If you need the lines use `read_string().lines()`.
    pub fn read_string(&self) -> Result<String, ResourceError> {
        let bytes = self.read_bytes()?;
        Ok(String::from_utf8(bytes)?)
    }

    /// Computes the BLAKE3 hash of the resource content.
    /// This is cached, so subsequent calls will not re-read the content.
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
        let resource = Resource::new(".")
            .unwrap()
            .with_git("HEAD".to_owned(), "./..")
            .expect("Repo must exist");

        let entries = resource.list().expect("Listing must succeed");
        assert!(!entries.is_empty(), "Directory should contain files");
    }

    #[test]
    fn test_git_file() {
        let resource = Resource::new("./src/main.rs")
            .unwrap()
            .with_git("HEAD".to_owned(), "./..")
            .expect("Repo must exist");

        assert!(resource.exists().unwrap(), "File should exist");
        let content = resource.read_string().expect("Should read file as string");
        let hash = resource.hash().expect("Should compute hash");

        assert!(!content.is_empty());
        assert_eq!(hash, blake3::hash(content.as_bytes()));
    }

    #[test]
    fn test_empty_resource() {
        let resource = Resource::empty();

        assert!(resource.exists().unwrap());
        assert_eq!(resource.read_bytes().unwrap(), b"");
        assert_eq!(resource.read_string().unwrap(), "");
        assert_eq!(resource.hash().unwrap(), blake3::hash(b""));
        assert_eq!(resource.list().unwrap(), Vec::<PathBuf>::new());
    }

    #[test]
    fn test_git_file_vs_worktree() {
        let git_resource = Resource::new("./src/main.rs")
            .unwrap()
            .with_git("HEAD".to_owned(), "./..")
            .unwrap();

        let wt_resource = Resource::new("./src/main.rs").unwrap();

        let git_content = git_resource.read_string().unwrap();
        let wt_content = wt_resource.read_string().unwrap();

        assert_eq!(git_content, wt_content);
    }

    #[test]
    fn test_missing_path() {
        let resource = Resource::new("does/not/exist.rs");

        assert!(
            resource.is_err(),
            "Resource creation should fail for non-existent path"
        );
    }
}
