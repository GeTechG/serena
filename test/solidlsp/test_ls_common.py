import logging
import os
from collections.abc import Sequence

import pytest

from solidlsp import SolidLanguageServer
from solidlsp.ls import DocumentSymbols, LSPFileBuffer
from solidlsp.ls_config import LanguageServerId
from solidlsp.ls_exceptions import SolidLSPException
from solidlsp.ls_utils import SymbolUtils
from test.conftest import PYTHON_LANGUAGE_BACKENDS, start_ls_context


class TestLanguageServerCommonFunctionality:
    """Test common functionality of SolidLanguageServer base implementation (not language-specific behaviour)."""

    @pytest.mark.parametrize("language_server", PYTHON_LANGUAGE_BACKENDS, indirect=True)
    def test_open_file_cache_invalidate(self, language_server: SolidLanguageServer) -> None:
        """
        Tests that the file buffer cache is invalidated when the file is changed on disk.
        """
        file_path = os.path.join(language_server.repository_root_path, "test_open_file.py")
        test_string1 = "# foo"
        test_string2 = "# bar"

        with open(file_path, "w") as f:
            f.write(test_string1)

        try:
            with language_server.open_file(file_path) as fb:
                assert fb.contents == test_string1

                # apply external change to file
                with open(file_path, "w") as f:
                    f.write(test_string2)

                # Explicitly bump mtime into the future so the cache sees a change.
                # Relying on natural mtime advancement is flaky because many filesystems
                # (ext4, tmpfs) have only 1-second mtime granularity, and both writes
                # can land in the same second.
                stat = os.stat(file_path)
                os.utime(file_path, (stat.st_atime, stat.st_mtime + 2))

                # check that the file buffer has been invalidated and reloaded
                assert fb.contents == test_string2

        finally:
            os.remove(file_path)

    def test_workspace_folders_affect_full_symbol_tree(self):
        """
        Tests that workspace folder configurations are respected when requesting the full symbol tree.
        """

        def check(langsrv: SolidLanguageServer, present: Sequence[str] = (), absent: Sequence[str] = ()) -> None:
            symbols = langsrv.request_full_symbol_tree()
            for name in present:
                assert SymbolUtils.symbol_tree_contains_name(symbols, name)
            for name in absent:
                assert not SymbolUtils.symbol_tree_contains_name(symbols, name)

        symbols_in_subfolder_test_repo = ["OuterClass", "UserService"]
        symbols_in_subfolder_scripts = ["parse_args", "create_sample_users"]
        all_symbols = symbols_in_subfolder_test_repo + symbols_in_subfolder_scripts

        with start_ls_context(ls_id=LanguageServerId.PYTHON, workspace_folders=["."]) as ls:
            ls.request_full_symbol_tree()
            check(ls, present=all_symbols)

        with start_ls_context(ls_id=LanguageServerId.PYTHON, workspace_folders=["./test_repo"]) as ls:
            ls.request_full_symbol_tree()
            check(ls, present=symbols_in_subfolder_test_repo, absent=symbols_in_subfolder_scripts)

        with start_ls_context(ls_id=LanguageServerId.PYTHON, workspace_folders=["./scripts"]) as ls:
            ls.request_full_symbol_tree()
            check(ls, present=symbols_in_subfolder_scripts, absent=symbols_in_subfolder_test_repo)

            # test that explicit requests for symbols in other workspace folders are rejected
            with pytest.raises(ValueError, match="outside"):
                ls.request_full_symbol_tree(within_relative_path="./test_repo")

    def test_full_symbol_tree_skips_file_without_document_symbols(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        Tests that a file for which the server fails to provide document symbols is left out of the full symbol tree
        instead of failing the whole traversal, while a request for symbols in that file alone still fails.
        """
        failing_file = os.path.join("test_repo", "models.py")

        with start_ls_context(ls_id=LanguageServerId.PYTHON) as ls:
            request_document_symbols = ls.request_document_symbols

            def request_document_symbols_failing_for_one_file(
                relative_file_path: str, file_buffer: LSPFileBuffer | None = None
            ) -> DocumentSymbols:
                if os.path.normpath(relative_file_path) == failing_file:
                    raise SolidLSPException("Unable to build token tree")
                return request_document_symbols(relative_file_path, file_buffer)

            monkeypatch.setattr(ls, "request_document_symbols", request_document_symbols_failing_for_one_file)

            with caplog.at_level(logging.WARNING, logger="solidlsp.ls"):
                symbols = ls.request_full_symbol_tree()
            assert SymbolUtils.symbol_tree_contains_name(symbols, "UserService")  # declared in test_repo/services.py
            assert not SymbolUtils.symbol_tree_contains_name(symbols, "BaseModel")  # declared in test_repo/models.py

            with pytest.raises(SolidLSPException):
                ls.request_full_symbol_tree(within_relative_path=failing_file)

        assert any("models.py" in r.getMessage() and "Unable to build token tree" in r.getMessage() for r in caplog.records)
