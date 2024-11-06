import { LevelUp, LevelUpChain } from 'levelup';
import { HashPath } from './hash_path';
import { Sha256Hasher } from './sha256_hasher';

const MAX_DEPTH = 32;
const LEAF_BYTES = 64; // All leaf values are 64 bytes.

/**
 * The merkle tree, in summary, is a data structure with a number of indexable elements, and the property
 * that it is possible to provide a succinct proof (HashPath) that a given piece of data, exists at a certain index,
 * for a given merkle tree root.
 */
export class MerkleTree {
  private hasher = new Sha256Hasher();
  private root = Buffer.alloc(32);
  private cache: Map<string, Buffer> = new Map();

  /**
   * Constructs a new MerkleTree instance, either initializing an empty tree, or restoring pre-existing state values.
   * Use the async static `new` function to construct.
   *
   * @param db Underlying leveldb.
   * @param name Name of the tree, to be used when restoring/persisting state.
   * @param depth The depth of the tree, to be no greater than MAX_DEPTH.
   * @param root When restoring, you need to provide the root.
   */
  constructor(private db: LevelUp, private name: string, private depth: number, root?: Buffer) {
    if (!(depth >= 1 && depth <= MAX_DEPTH)) {
      throw Error('Bad depth');
    }

    this.root = root || this.getEmptyNodeHash(depth);
  }

  /**
   * Constructs or restores a new MerkleTree instance with the given `name` and `depth`.
   * The `db` contains the tree data.
   */
  static async new(db: LevelUp, name: string, depth = MAX_DEPTH) {
    const meta: Buffer = await db.get(Buffer.from(name)).catch(() => {});
    if (meta) {
      const root = meta.slice(0, 32);
      const depth = meta.readUInt32LE(32);
      return new MerkleTree(db, name, depth, root);
    } else {
      const tree = new MerkleTree(db, name, depth);
      await tree.writeMetaData();
      return tree;
    }
  }

  private async writeMetaData(batch?: LevelUpChain<string, Buffer>) {
    const data = Buffer.alloc(40);
    this.root.copy(data);
    data.writeUInt32LE(this.depth, 32);
    if (batch) {
      batch.put(this.name, data);
    } else {
      await this.db.put(this.name, data);
    }
  }

  getRoot() {
    return this.root;
  }

  /**
   * Returns the hash path for `index`.
   * e.g. To return the HashPath for index 2, return the nodes marked `*` at each layer.
   *     d3:                                            [ root ]
   *     d2:                      [*]                                               [*]
   *     d1:         [*]                      [*]                       [ ]                     [ ]
   *     d0:   [ ]         [ ]          [*]         [*]           [ ]         [ ]          [ ]        [ ]
   */
  async getHashPath(index: number): Promise<HashPath> {
    const path = new HashPath();
    let currentIndex = index;

    for (let height = 0; height < this.depth; height++) {
      const isRight = currentIndex % 2 === 1;
      const pairIndex = isRight ? currentIndex - 1 : currentIndex + 1;
      const pairHash = await this.getNode(pairIndex, height);
      const currentHash = await this.getNode(currentIndex, height);
      path.data.push(isRight ? [pairHash, currentHash] : [currentHash, pairHash]);
      currentIndex = Math.floor(currentIndex / 2);
    }

    return path;
  }

  /**
   * Updates the tree with `value` at `index`. Returns the new tree root.
   */
  async updateElement(index: number, value: Buffer): Promise<Buffer> {
    const batch = this.db.batch();
    let currentValue = this.hasher.hash(value);
    let currentIndex = index;

    // Store the leaf
    await this.updateNode(currentIndex, 0, currentValue, batch);

    // Update path to root
    for (let height = 0; height < this.depth; height++) {
      const isRight = currentIndex % 2 === 1;
      const pairIndex = isRight ? currentIndex - 1 : currentIndex + 1;
      const pairHash = await this.getNode(pairIndex, height);

      currentValue = isRight 
        ? this.hasher.compress(pairHash, currentValue)
        : this.hasher.compress(currentValue, pairHash);

      currentIndex = Math.floor(currentIndex / 2);
      await this.updateNode(currentIndex, height + 1, currentValue, batch);
    }

    this.root = currentValue;
    await this.writeMetaData(batch);
    await batch.write();
    return this.root;
  }

  /**
   * Updates multiple elements in the tree in a single batch operation.
   * @param updates Array of [index, value] tuples to update
   * @returns The new tree root
   */
  async updateElements(updates: Array<[number, Buffer]>): Promise<Buffer> {
    if (updates.length === 0) return this.root;
    
    const batch = this.db.batch();
    const nodeUpdates = new Map<string, Buffer>();
    
    // First, update all leaves
    for (const [index, value] of updates) {
      const leafValue = this.hasher.hash(value);
      await this.updateNode(index, 0, leafValue, batch);
      nodeUpdates.set(`0:${index}`, leafValue);
    }

    // Then update all affected parent nodes level by level
    for (let height = 0; height < this.depth; height++) {
      const currentLevelUpdates = new Map<number, Buffer>();
      
      // Collect all indices that need updating at this height
      const indices = new Set(
        Array.from(nodeUpdates.keys())
          .filter(key => key.startsWith(`${height}:`))
          .map(key => parseInt(key.split(':')[1]))
      );

      // Update each affected node's parent
      for (const index of indices) {
        const parentIndex = Math.floor(index / 2);
        const isRight = index % 2 === 1;
        const siblingIndex = isRight ? index - 1 : index + 1;
        
        const currentValue = nodeUpdates.get(`${height}:${index}`) || 
          await this.getNode(index, height);
        const siblingValue = nodeUpdates.get(`${height}:${siblingIndex}`) || 
          await this.getNode(siblingIndex, height);
        
        const parentValue = isRight
          ? this.hasher.compress(siblingValue, currentValue)
          : this.hasher.compress(currentValue, siblingValue);
        
        await this.updateNode(parentIndex, height + 1, parentValue, batch);
        currentLevelUpdates.set(parentIndex, parentValue);
      }

      // Add current level updates to nodeUpdates for next iteration
      for (const [index, value] of currentLevelUpdates) {
        nodeUpdates.set(`${height + 1}:${index}`, value);
      }
    }

    // Update root and metadata
    this.root = nodeUpdates.get(`${this.depth}:0`) || this.root;
    await this.writeMetaData(batch);
    await batch.write();
    return this.root;
  }

  private getEmptyNodeHash(height: number): Buffer {
    if (height === 0) {
      // Leaf level - hash 64 zero bytes
      return this.hasher.hash(Buffer.alloc(LEAF_BYTES));
    }
    // Internal node - compress two identical child empty hashes
    const childHash = this.getEmptyNodeHash(height - 1);
    return this.hasher.compress(childHash, childHash);
  }

  private async getNode(index: number, height: number): Promise<Buffer> {
    const key = `${this.name}/node/${height}/${index}`;
    
    // Check cache first
    const cached = this.cache.get(key);
    if (cached) return cached;

    try {
      const value = await this.db.get(Buffer.from(key));
      this.cache.set(key, value);
      return value;
    } catch {
      const empty = this.getEmptyNodeHash(height);
      this.cache.set(key, empty);
      return empty;
    }
  }

  private async updateNode(index: number, height: number, value: Buffer, batch: LevelUpChain<string, Buffer>) {
    const key = `${this.name}/node/${height}/${index}`;
    this.cache.set(key, value);
    batch.put(key, value);
  }
}
