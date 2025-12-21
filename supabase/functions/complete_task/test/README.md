# complete_task Test Suite

This test suite validates the `complete_task` edge function behavior to ensure refactoring doesn't break functionality.

## Refactoring Validation Workflow

### Step 1: Run tests BEFORE refactoring
```bash
./test/run_all.sh
```
This establishes the baseline behavior (golden files already created).

### Step 2: Do the refactoring
Extract modules as planned (e.g., `params.ts`, `generation.ts`, `storage.ts`, etc.)

### Step 3: Update test imports
In `unit_tests_refactored.ts`:
1. Uncomment the imports from your new modules
2. Delete the temporary local function copies
3. Run: `npx deno test --no-check test/unit_tests_refactored.ts`

### Step 4: Run all tests AFTER refactoring
```bash
./test/run_all.sh
```
If tests pass → refactoring preserved behavior ✓

---

## Test Types

### 1. Unit Tests (`unit_tests.ts`)
Tests pure functions in isolation:
- `getContentType()` - MIME type detection
- `extractFromParams()` - Nested param extraction
- `extractOrchestratorTaskId()` - Orchestrator task ID extraction
- `extractOrchestratorRunId()` - Run ID extraction
- `extractBasedOn()` - Based-on generation extraction
- `extractShotAndPosition()` - Shot ID and position flag extraction
- `setThumbnailInParams()` - Thumbnail URL placement by task type
- `buildGenerationParams()` - Generation param construction

### 2. Scenario Tests (`runner.ts`)
Tests full handler behavior through DB/storage operation capture:
- **MODE 1**: Base64 upload (simple image, video with thumbnail)
- **MODE 3**: Pre-signed URL reference (with/without thumbnail)
- **Inpaint**: Creates variant on source generation
- **Upscale**: Creates primary variant
- **Travel segment**: Creates child generation + parent variant
- **Travel stitch**: Updates parent generation
- **Error cases**: Missing fields, invalid base64

## Running Tests

```bash
# Quick run
cd supabase/functions/complete_task
./test/run_all.sh

# Or manually:
npx deno test --no-check --allow-read --allow-env test/unit_tests.ts
npx deno run --allow-read --allow-write --allow-env test/runner.ts
```

## Golden Files

The `golden/` directory contains snapshots of expected behavior:
- `results.json` - Captured DB/storage operations for each scenario

### Updating Golden Files

After intentional behavior changes, update the golden files:

```bash
./test/run_all.sh --update
# or
npx deno run --allow-read --allow-write --allow-env test/runner.ts --update
```

## Refactoring Workflow

1. **Before refactoring**: Run tests to ensure they pass
2. **After refactoring**: Run tests without `--update`
3. **If tests fail**: Either fix the code or update golden files if change was intentional

## Test Fixtures (`fixtures.ts`)

Contains:
- `IDS` - Consistent UUIDs for test data
- `baseMockConfig` - Mock database state
- `TEST_SCENARIOS` - All test scenarios with expected behavior

## Adding New Tests

1. Add scenario to `TEST_SCENARIOS` in `fixtures.ts`
2. Add mock data if needed in `baseMockConfig`
3. Run with `--update` to capture expected behavior
4. Commit the updated golden files

## Mock Infrastructure (`mocks.ts`)

- `OperationCapture` - Captures all DB/storage operations
- `createMockSupabase()` - Mock Supabase client
- `createMockFetch()` - Mock external API calls
- `createMockRequest()` - Create test HTTP requests






